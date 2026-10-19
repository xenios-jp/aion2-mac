/* F18 and F19 (spec fg.interm_motion section 7): im-time, im-nearest, im-radius, im-border, im-weights, im-sigmoid, im-random.
   A CPU reference of section 3 in IEEE fp32 is compared with the engine on each scene, the explicit expectations of the spec
   are checked on top, and the controls are mutations of the reference that must differ from the engine's output. */
#import <Foundation/Foundation.h>
#pragma clang fp contract(off)
#include "texutil.h"
#include "chain.h"
#include "probe.h"

static int fails;
static NnRig R;
static MetalFGChain *CH;

enum { X_NONE = 0, X_HALF, X_SWAP, X_NOSPREAD, X_PAYREV, X_SPMIN, X_INGRID, X_ZEROOUT, X_FARTEST, X_NOTRUNC, X_FARWINS, X_SPMAX, X_ONE, X_NOCLOSE, X_ZNPREV, X_EMPTYGE };

typedef struct {
    int Wm, Hm, Wf, Hf;
    float t, H, d_sim, th_zn, Zsep, Zs;
    int sp_min, sp_max;
    float *mvCx, *mvCy, *mvPx, *mvPy;    /* Mot pixels (C is converted to normalised units for the texture) */
    float *dC, *dP;
    float *lgPx, *lgPy, *lgCx, *lgCy;
} S18;

typedef struct {
    uint32_t *ic, *ip;                  /* key, payload words: key = 0xFFFFFFFF when empty */
    float *wp, *wc;
} O18;

static float lamb(const S18 *s, float d) { return s->Zs * (1.0f / fmaxf(1.0f - d, 1e-7f)); }
static uint16_t hb(float v) { h16 h = (h16)v; uint16_t b; memcpy(&b, &h, 2); return b; }
static float sg(float v) { return v > 0 ? 1.0f : -1.0f; }

static void scatter_ref(const S18 *s, int phase, int mut, uint32_t *key, uint32_t *pay) {
    int W = s->Wm, H = s->Hm;
    float f = phase == 0 ? (1.0f - s->t) : s->t;
    if (mut == X_HALF) f = 0.5f;
    if (mut == X_SWAP) f = phase == 0 ? s->t : (1.0f - s->t);
    const float *fx = phase == 0 ? s->mvCx : s->mvPx, *fy = phase == 0 ? s->mvCy : s->mvPy, *dd = phase == 0 ? s->dC : s->dP;
    float sx = phase == 0 ? f * (float)W : f, sy = phase == 0 ? f * (float)H : f;
    for (int i = 0; i < W * H; i++) key[i] = pay[i] = 0xFFFFFFFFu;
#define CL(v, n) ((v) < 0 ? 0 : (v) >= (n) ? (n) - 1 : (v))
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int pi = y * W + x;
            float hx = fx[pi] * (phase == 0 ? sx / (float)W : 1.0f) * 1.0f, hy = fy[pi] * (phase == 0 ? sy / (float)H : 1.0f);
            /* h = field * scale: the engine multiplies the normalised field by f * Wm; the reference holds Mot pixels */
            hx = phase == 0 ? fx[pi] * f : fx[pi] * f; hy = phase == 0 ? fy[pi] * f : fy[pi] * f;
            float kp = lamb(s, dd[pi]);
            float mdx = 0, mdy = 0, spread = 0;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                int qx = x + dx, qy = y + dy;
                int out = qx < 0 || qy < 0 || qx >= W || qy >= H;
                if (out && mut == X_INGRID) continue;
                int cx = CL(qx, W), cy = CL(qy, H), qi = cy * W + cx;
                float qhx = fx[qi] * f, qhy = fy[qi] * f, kq = lamb(s, dd[qi]);
                if (out && mut == X_ZEROOUT) { qhx = qhy = 0; kq = lamb(s, 0.0f); }
                mdx = fmaxf(mdx, fabsf((float)dx + qhx - hx)); mdy = fmaxf(mdy, fabsf((float)dy + qhy - hy));
                spread = fmaxf(spread, fabsf(kq - kp));
            }
            int rx, ry;
            if (spread > s->Zsep && mut != X_NOSPREAD) rx = ry = s->sp_min;
            else {
                rx = (int)fminf(fmaxf(ceilf(mdx + 0.5f), (float)s->sp_min), (float)s->sp_max);
                ry = (int)fminf(fmaxf(ceilf(mdy + 0.5f), (float)s->sp_min), (float)s->sp_max);
            }
            if (mut == X_SPMIN) rx = ry = s->sp_min;
            if (mut == X_SPMAX) rx = ry = s->sp_max;
            if (mut == X_ONE) rx = ry = 1;
            int ssx = (int)sg(hx), ssy = (int)sg(hy), t0x = (int)truncf(hx), t0y = (int)truncf(hy);
            float thr = fmaxf(1.0f, (hx * hx + hy * hy) * s->d_sim / (s->H * s->H));
            float ox = -hx / (float)W, oy = -hy / (float)H;
            uint32_t kb; memcpy(&kb, &kp, 4);
            uint32_t pl = ((uint32_t)hb(ox) << 16) | hb(oy);
            if (mut == X_PAYREV) pl = ~pl;
            for (int j = 0; j < ry; j++) for (int i = 0; i < rx; i++) {
                int qx = x + i * ssx, qy = y + j * ssy, Tx = qx + t0x, Ty = qy + t0y;
                if (Tx < 0 || Ty < 0 || Tx >= W || Ty >= H || qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
                int take = (i == 0 && j == 0);
                int qi = qy * W + qx;
                if (!take) { float ex = fx[qi] * f - hx, ey = fy[qi] * f - hy; take = ex * ex + ey * ey < thr; }
                if (!take && !(phase == 0 && mut == X_NOCLOSE) && (phase == 0 || mut == X_ZNPREV)) {   /* rule C: the previous-call projection has no scale-distance alternative */
                    float kq = lamb(s, dd[qi]);
                    float tq = mut == X_NOTRUNC ? kq : fminf(truncf(kq), 4294967295.0f);
                    take = fabsf(tq - kp) < s->th_zn;
                }
                if (!take) continue;
                int c = Ty * W + Tx;
                int better = mut == X_FARWINS ? (key[c] == 0xFFFFFFFFu || kb > key[c] || (kb == key[c] && pl < pay[c])) : (kb < key[c] || (kb == key[c] && pl < pay[c]));
                if (better) { key[c] = kb; pay[c] = pl; }
            }
        }
}

static float sigf(float v) { return v >= 0 ? 1.0f / (1.0f + expf(-v)) : expf(v) / (1.0f + expf(v)); }

static float bil2(const float *img, int W, int H, float u, float v) {
    float x = u * W - 0.5f, y = v * H - 0.5f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
#define AT(X, Y) img[CL(Y, H) * W + CL(X, W)]
    return (AT(x0, y0) * (1 - fx) + AT(x0 + 1, y0) * fx) * (1 - fy) + (AT(x0, y0 + 1) * (1 - fx) + AT(x0 + 1, y0 + 1) * fx) * fy;
}

static void dec(uint32_t key, uint32_t pay, float *ox, float *oy, int *empty) {
    *empty = key == 0xFFFFFFFFu;
    *ox = *oy = 0;
    if (*empty) return;
    uint16_t bx = pay >> 16, by = pay & 0xFFFF;
    h16 hx, hy; memcpy(&hx, &bx, 2); memcpy(&hy, &by, 2);
    *ox = (float)hx; *oy = (float)hy;
}

static void weights_ref(const S18 *s, int mut, const uint32_t *kc, const uint32_t *pc, const uint32_t *kp, const uint32_t *pp, float *wp, float *wc) {
    int W = s->Wm, H = s->Hm;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        int i = y * W + x, e1, e2;
        float oCx, oCy, oPx, oPy;
        dec(kc[i], pc[i], &oCx, &oCy, &e1); dec(kp[i], pp[i], &oPx, &oPy, &e2);
        float ux = ((float)x + 0.5f) / W, uy = ((float)y + 0.5f) / H;
        int fx = (int)ceilf(ux * s->Wf) - 1, fy = (int)ceilf(uy * s->Hf) - 1;
        fx = CL(fx, s->Wf); fy = CL(fy, s->Hf);
        float xp = bil2(s->lgPx, s->Wf, s->Hf, ux + oPx, uy + oPy), xc = bil2(s->lgCx, s->Wf, s->Hf, ux + oCx, uy + oCy);
        (void)mut;
        wp[i] = sigf(xp - s->lgPy[fy * s->Wf + fx]);
        wc[i] = sigf(xc - s->lgCy[fy * s->Wf + fx]);
    }
}

static S18 *new_scene(int W, int H, int Wf, int Hf) {
    S18 *s = calloc(1, sizeof *s);
    s->Wm = W; s->Hm = H; s->Wf = Wf; s->Hf = Hf; s->t = 0.5f; s->H = 360; s->d_sim = 10000; s->th_zn = 4; s->Zsep = 0.5f; s->Zs = 4;
    s->sp_min = 2; s->sp_max = 4;
    size_t n = (size_t)W * H, nf = (size_t)Wf * Hf;
    s->mvCx = calloc(n, 4); s->mvCy = calloc(n, 4); s->mvPx = calloc(n, 4); s->mvPy = calloc(n, 4);
    s->dC = malloc(n * 4); s->dP = malloc(n * 4);
    for (size_t i = 0; i < n; i++) s->dC[i] = s->dP[i] = 63.0f / 64;
    s->lgPx = calloc(nf, 4); s->lgPy = calloc(nf, 4); s->lgCx = calloc(nf, 4); s->lgCy = calloc(nf, 4);
    return s;
}

static uint32_t depth_word(float d) { return rec_pack_c(d, 0); }

static void run_engine(const S18 *s, O18 *o) {
    int W = s->Wm, H = s->Hm, Wf = s->Wf, Hf = s->Hf;
    id<MTLTexture> fC = gpu_tex(&R.gpu, MTLPixelFormatRG32Float, W, H), fP = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, W, H);
    id<MTLTexture> dC = gpu_tex(&R.gpu, MTLPixelFormatR32Uint, W, H), dP = gpu_tex(&R.gpu, MTLPixelFormatR32Uint, W, H);
    id<MTLTexture> lgP = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, Wf, Hf), lgC = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, Wf, Hf);
    id<MTLTexture> bw = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, W, H);
    tex_set(fC, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->mvCx[y * W + x] / W; v[1] = s->mvCy[y * W + x] / H; });
    tex_set(fP, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->mvPx[y * W + x]; v[1] = s->mvPy[y * W + x]; });
    uint32_t *words = malloc((size_t)W * H * 4);
    for (int i = 0; i < W * H; i++) words[i] = depth_word(s->dC[i]);
    gpu_upload(dC, words, 4);
    for (int i = 0; i < W * H; i++) words[i] = depth_word(s->dP[i]);
    gpu_upload(dP, words, 4);
    free(words);
    tex_set(lgP, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->lgPx[y * Wf + x]; v[1] = s->lgPy[y * Wf + x]; });
    tex_set(lgC, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = s->lgCx[y * Wf + x]; v[1] = s->lgCy[y * Wf + x]; });
    tex_set(bw, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = NAN; });
    id<MTLBuffer> sc[2];
    for (int k = 0; k < 2; k++) {
        sc[k] = [R.gpu.dev newBufferWithLength:(size_t)W * H * 8 options:MTLResourceStorageModeShared];
        uint32_t *p = sc[k].contents;
        for (int i = 0; i < W * H * 2; i++) p[i] = 0xFFFFFFFFu;
    }
    MetalFGF18Args a;
    memset(&a, 0, sizeof a);
    a.Wm = W; a.Hm = H; a.Wf = Wf; a.Hf = Hf; a.H = s->H; a.d_sim = s->d_sim; a.th_zn = s->th_zn; a.Z_sep = s->Zsep;
    a.sp_min = s->sp_min; a.sp_max = s->sp_max; a.D = (MetalFGDepthConst){s->Zs, 1024, s->Zsep, 1e-7f, 0};
    float omt = 1.0f - s->t;
    static float hsC[2], hsP[2];
    hsC[0] = omt * W; hsC[1] = omt * H; hsP[0] = hsP[1] = s->t;
    id<MTLBuffer> sc0 = sc[0], sc1 = sc[1];
    nn_run(&R, ^(MetalFGEnc *e) {
        [CH f18:e args:a field:fC hscale:hsC dep:dC sc:sc0 useZn:YES];
        [CH f18:e args:a field:fP hscale:hsP dep:dP sc:sc1 useZn:NO];
        [CH f19:e args:a scP:sc1 scC:sc0 lgP:lgP lgC:lgC bw:bw];
    });
    uint32_t *c = sc[0].contents, *p = sc[1].contents;
    for (int i = 0; i < W * H; i++) { o->ic[i * 2] = c[2 * i + 1]; o->ic[i * 2 + 1] = c[2 * i]; o->ip[i * 2] = p[2 * i + 1]; o->ip[i * 2 + 1] = p[2 * i]; }
    float *w = tex_get(bw, 2);
    for (int i = 0; i < W * H; i++) { o->wp[i] = w[2 * i]; o->wc[i] = w[2 * i + 1]; }
    free(w);
}

static O18 *new_out(int n) {
    O18 *o = calloc(1, sizeof *o);
    o->ic = calloc(n * 2, 4); o->ip = calloc(n * 2, 4); o->wp = calloc(n, 4); o->wc = calloc(n, 4);
    return o;
}

/* number of cells where the decoded views or the weights differ */
static int cmp(const S18 *s, uint32_t *kc, uint32_t *pc, uint32_t *kp, uint32_t *pp, const O18 *g, float *wp, float *wc, int *excl) {
    int W = s->Wm, H = s->Hm, bad = 0;
    for (int i = 0; i < W * H; i++) {
        int e1, e2, e3, e4;
        float a1, a2, b1, b2, c1, c2, d1, d2;
        dec(g->ic[2 * i], g->ic[2 * i + 1], &a1, &a2, &e1); dec(kc[i], pc[i], &b1, &b2, &e2);
        dec(g->ip[2 * i], g->ip[2 * i + 1], &c1, &c2, &e3); dec(kp[i], pp[i], &d1, &d2, &e4);
        if (excl && excl[i]) continue;
        bad += e1 != e2 || e3 != e4 || fabsf(a1 - b1) * W > 0x1p-8 || fabsf(a2 - b2) * H > 0x1p-8 || fabsf(c1 - d1) * W > 0x1p-8 || fabsf(c2 - d2) * H > 0x1p-8;
        bad += fabsf(g->wp[i] - wp[i]) > 0x1p-8 || fabsf(g->wc[i] - wc[i]) > 0x1p-8 || isnan(g->wp[i]);
    }
    return bad;
}

static void judge(const char *name, const S18 *s, const int *ctl, int nctl, int max_bad, int check_weights) {
    int n = s->Wm * s->Hm;
    O18 *g = new_out(n);
    run_engine(s, g);
    uint32_t *kc = malloc(n * 4), *pc = malloc(n * 4), *kp = malloc(n * 4), *pp = malloc(n * 4);
    float *wp = malloc(n * 4), *wc = malloc(n * 4);
    scatter_ref(s, 0, 0, kc, pc); scatter_ref(s, 1, 0, kp, pp);
    weights_ref(s, 0, kc, pc, kp, pp, wp, wc);
    if (!check_weights) for (int i = 0; i < n; i++) { g->wp[i] = wp[i]; g->wc[i] = wc[i]; }
    int bad = cmp(s, kc, pc, kp, pp, g, wp, wc, NULL);
    CHECKC(bad <= max_bad, "%s: %d cells differ from the reference (allowed %d)", name, bad, max_bad);
    for (int i = 0; i < nctl; i++) {
        uint32_t *kc2 = malloc(n * 4), *pc2 = malloc(n * 4), *kp2 = malloc(n * 4), *pp2 = malloc(n * 4);
        scatter_ref(s, 0, ctl[i], kc2, pc2); scatter_ref(s, 1, ctl[i], kp2, pp2);
        int d = cmp(s, kc2, pc2, kp2, pp2, g, wp, wc, NULL);
        if (ctl[i] == X_INGRID || ctl[i] == X_SPMIN) { if (d < 10) printf("  %s: control %d is void (%d cells)\n", name, ctl[i], d); }
        else CHECKC(d >= (max_bad ? (ctl[i] == X_ZEROOUT ? 4 : 10) : 1), "%s: control %d differs at only %d cells", name, ctl[i], d);
        free(kc2); free(pc2); free(kp2); free(pp2);
    }
    free(kc); free(pc); free(kp); free(pp); free(wp); free(wc);
    free(g->ic); free(g->ip); free(g->wp); free(g->wc); free(g);
}

#define W_ 320
#define H_ 180
#define ID(x, y) ((y) * W_ + (x))

static void fill_block(S18 *s, int x0, int x1, int y0, int y1, float d, float cx, float cy) {
    for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) { s->dC[ID(x, y)] = s->dP[ID(x, y)] = d; s->mvCx[ID(x, y)] = cx; s->mvCy[ID(x, y)] = cy; }
}

static void cases(void) {
    O18 *g;
    int n = W_ * H_;
    /* im-time */
    {
        S18 *s = new_scene(W_, H_, W_, H_);
        s->t = 0.25f;
        for (int i = 0; i < n; i++) { s->dC[i] = s->dP[i] = 0.5f; s->mvCx[i] = -10; s->mvPx[i] = 10; }
        int ctl[] = {X_HALF, X_SWAP};
        judge("im-time", s, ctl, 2, 0, 0);
        g = new_out(n); run_engine(s, g);
        int bad = 0;
        for (int y = 0; y < H_; y++) for (int x = 0; x < W_; x++) {
            float ox, oy; int e;
            dec(g->ic[2 * ID(x, y)], g->ic[2 * ID(x, y) + 1], &ox, &oy, &e);
            bad += e != (x >= 313) || (!e && fabsf(ox * W_ - 7.5f) > 0x1p-8);
            dec(g->ip[2 * ID(x, y)], g->ip[2 * ID(x, y) + 1], &ox, &oy, &e);
            bad += e != (x <= 1) || (!e && fabsf(ox * W_ + 2.5f) > 0x1p-8);
        }
        CHECKC(bad == 0, "im-time explicit: %d", bad);
        free(g);
        free(s);
    }
    /* im-nearest */
    {
        S18 *s = new_scene(W_, H_, W_, H_);
        s->dC[ID(30, 20)] = s->dC[ID(34, 20)] = s->dC[ID(30, 24)] = 0.5f;
        s->mvCx[ID(30, 20)] = 5; s->mvCx[ID(34, 20)] = -5; s->mvCx[ID(30, 24)] = 5; s->mvCx[ID(34, 24)] = -5;
        s->dC[ID(34, 24)] = 9.0f / 16;
        int ctl[] = {X_PAYREV, X_FARWINS};
        judge("im-nearest", s, ctl, 2, 0, 0);
        g = new_out(n); run_engine(s, g);
        float ox, oy; int e;
        dec(g->ic[2 * ID(32, 20)], g->ic[2 * ID(32, 20) + 1], &ox, &oy, &e);
        CHECKC(!e && fabsf(ox * W_ - 2.5f) < 0x1p-8, "im-nearest (32,20): %g", ox * W_);
        dec(g->ic[2 * ID(32, 24)], g->ic[2 * ID(32, 24) + 1], &ox, &oy, &e);
        CHECKC(!e && fabsf(ox * W_ + 2.5f) < 0x1p-8, "im-nearest (32,24): %g", ox * W_);
        free(g); free(s);
    }
    /* im-radius */
    {
        S18 *s = new_scene(W_, H_, W_, H_);
        for (int i = 0; i < n; i++) s->dC[i] = s->dP[i] = 9.0f / 16;
        fill_block(s, 100, 115, 60, 75, 0.5f, 5, 0);
        int ctl[] = {X_ONE, X_NOCLOSE, X_SPMAX};
        judge("im-radius", s, ctl, 3, 0, 0);
        g = new_out(n); run_engine(s, g);
        int bad = 0;
        for (int y = 0; y < H_; y++) for (int x = 0; x < W_; x++) {
            float ox, oy; int e;
            dec(g->ic[2 * ID(x, y)], g->ic[2 * ID(x, y) + 1], &ox, &oy, &e);
            if (y == 75 && (x == 100 || x == 101)) continue;
            int inrect = x >= 102 && x <= 118 && y >= 59 && y <= 75;
            int vacated = x >= 100 && x <= 101 && y >= 60 && y <= 74;
            if (inrect) bad += e || fabsf(ox * W_ + 2.5f) > 0x1p-8;
            else if (vacated) bad += !e;
            else bad += e || ox != 0;
        }
        CHECKC(bad == 0, "im-radius explicit: %d", bad);
        free(g); free(s);
    }
    /* im-border */
    {
        S18 *s = new_scene(W_, H_, W_, H_);
        for (int y = 0; y < H_; y++) for (int x = 0; x < 4; x++) { s->dC[ID(x, y)] = s->dP[ID(x, y)] = 0; s->mvCx[ID(x, y)] = 5; s->mvCy[ID(x, y)] = x == 0 ? 0 : 0.5f; }
        int ctl[] = {X_ZEROOUT};
        judge("im-border", s, ctl, 1, 0, 0);
        g = new_out(n); run_engine(s, g);
        int bad = 0;
        for (int y = 0; y < H_; y++) {
            float ox, oy; int e;
            dec(g->ic[2 * ID(4, y)], g->ic[2 * ID(4, y) + 1], &ox, &oy, &e);
            bad += e || fabsf(ox * W_ + 2.5f) > 0x1p-8 || fabsf(oy * H_ + 0.25f) > 0x1p-8;
        }
        CHECKC(bad == 0, "im-border explicit: %d", bad);
        free(g); free(s);
    }
    /* im-weights, im-sigmoid */
    for (int variant = 0; variant < 3; variant++) {
        S18 *s = new_scene(32, 24, 16, 12);
        for (int i = 0; i < 32 * 24; i++) { s->dC[i] = s->dP[i] = 0.5f; s->mvCx[i] = -13; s->mvPx[i] = 13; }
        for (int y = 0; y < 12; y++) for (int x = 0; x < 16; x++) {
            float v = (x + 0.5f) / 16;
            int i = y * 16 + x;
            if (variant == 0) { s->lgPx[i] = s->lgCx[i] = 4 * v; s->lgPy[i] = s->lgCy[i] = 0.5f; }
            else if (variant == 1) { s->lgPx[i] = 100; s->lgPy[i] = -100; s->lgCx[i] = -100; s->lgCy[i] = 100; }
            else { s->lgPx[i] = 0; s->lgPy[i] = 4 * v; s->lgCx[i] = -100; s->lgCy[i] = 100; }
        }
        O18 *o = new_out(32 * 24);
        run_engine(s, o);
        int bad = 0, nan = 0;
        for (int y = 0; y < 24; y++) for (int x = 0; x < 32; x++) {
            double wp, wc;
            if (variant == 0) {
                wp = (x == 0 || x == 6) ? 1 / (1 + exp(3.0 / 8)) : x <= 5 ? 1 / (1 + exp(-(2.0 * x - 7) / 16)) : 1 / (1 + exp(-(x - 10) / 8.0));
                wc = x <= 24 ? 1 / (1 + exp(-(x + 3) / 8.0)) : (x == 25 || x == 31) ? 1 / (1 + exp(-27 / 8.0)) : 1 / (1 + exp(-(2.0 * x - 7) / 16));
            } else if (variant == 1) { wp = 1; wc = 0; }
            else { wp = 1 / (1 + exp((2 * floor(x / 2.0) + 1) / 8)); wc = 0; }
            int i = y * 32 + x;
            nan += isnan(o->wp[i]) || isnan(o->wc[i]);
            bad += fabs(o->wp[i] - wp) > 0x1p-8 || fabs(o->wc[i] - wc) > 0x1p-8;
        }
        CHECKC(bad == 0 && nan == 0, "im-%s variant %d: %d wrong, %d NaN", variant == 0 ? "weights" : "sigmoid", variant, bad, nan);
        if (variant == 0) judge("im-weights (reference)", s, NULL, 0, 0, 1);
        free(o);
        free(s);
    }
    /* im-random */
    for (int run = 0; run < 3; run++) {
        S18 *s = new_scene(W_, H_, 160, 90);
        s->t = run == 0 ? 0.25f : run == 1 ? 0.5f : 0.75f;
        uint64_t st = 2;
        const float dset[3] = {0.5f, 9.0f / 16, 63.0f / 64}, cset[4] = {-5, 0, 1.5f, 4.5f}, pset[4] = {-4, 0, 1.5f, 3};
#define NEXT() (st = st * 6364136223846793005ull + 1442695040888963407ull, (uint32_t)(st >> 33))
        for (int ty = 0; ty < (H_ + 7) / 8; ty++) for (int tx = 0; tx < W_ / 8; tx++) {
            float d = dset[NEXT() % 3], cx = cset[NEXT() & 3], cy = cset[NEXT() & 3], px = pset[NEXT() & 3], py = pset[NEXT() & 3];
            for (int y = ty * 8; y < ty * 8 + 8 && y < H_; y++) for (int x = tx * 8; x < tx * 8 + 8; x++) {
                int i = ID(x, y);
                s->dC[i] = s->dP[i] = d; s->mvCx[i] = cx; s->mvCy[i] = cy; s->mvPx[i] = px; s->mvPy[i] = py;
            }
        }
        const float sl[3] = {-2, 0, 2};
        for (int k = 0; k < 160 * 90; k++) {
            int x = k % 160, y = k / 160;
            s->lgPx[k] = sl[NEXT() % 3] * (x + y) / 160; s->lgPy[k] = sl[NEXT() % 3] * x / 160;
            s->lgCx[k] = 0.5f * x / 160; s->lgCy[k] = -0.25f * y / 90;
        }
        int ctl[] = {X_NOSPREAD, X_PAYREV, X_SPMIN, X_ZEROOUT};
        char nm[32];
        snprintf(nm, sizeof nm, "im-random t=%.2f", s->t);
        judge(nm, s, ctl, 4, n * 2 / 100, 1);
        free(s);
    }
}

/* ---- fg.interm_motion_pin_r4: im2-lowbits, im2-record-range, im2-key-once, im2-prev-rim, im2-bg-key ---- */

static float key_f(uint32_t k) { float f; memcpy(&f, &k, 4); return f; }
static float f_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static const float KB = 0x1p-21f;       /* relative error bound of the reciprocal */

static void im2_lowbits(void) {
    S18 *s = new_scene(W_, H_, W_, H_);
    const float dbg = f_bits(0x3F100005u), dob = f_bits(0x3F000007u);
    for (int i = 0; i < W_ * H_; i++) s->dC[i] = s->dP[i] = dbg;
    fill_block(s, 100, 115, 60, 75, dob, 0, 0);
    int n = W_ * H_;
    O18 *g = new_out(n);
    run_engine(s, g);
    uint32_t *kc = malloc(n * 4), *pc = malloc(n * 4);
    scatter_ref(s, 0, 0, kc, pc);
    int bad = 0;
    for (int i = 0; i < n; i++) {
        float ke = key_f(g->ic[2 * i]), km = key_f(kc[i]);
        bad += g->ic[2 * i] == 0xFFFFFFFFu || fabsf(ke - km) > KB * km;
    }
    CHECKC(bad == 0, "im2-lowbits: %d cells with a key outside the bound of the model's scale", bad);
    /* control: a record that overwrites the three lowest mantissa bits with the selector (they read as zero here) */
    float dcl = f_bits(0x3F000007u & ~7u), kctl = lamb(s, dcl), km = lamb(s, dob), ke = key_f(g->ic[2 * (68 * W_ + 108)]);
    CHECKC(fabsf(kctl - km) > KB * km && fabsf(ke - kctl) > KB * kctl, "im2-lowbits control: the cleared-bits scale %g is not outside the bound of the engine's %g (model %g)", kctl, ke, km);
    free(kc); free(pc); free(g->ic); free(g->ip); free(g->wp); free(g->wc); free(g); free(s);
}

static void im2_record_range(void) {
    Gpu *gp = &R.gpu;
    NSString *hdr = [NSString stringWithContentsOfFile:@"shaders/fgcommon.h" encoding:NSUTF8StringEncoding error:nil];
    if (!hdr) { printf("shaders/fgcommon.h not found (run from the repository root)\n"); exit(2); }
    NSString *src = [hdr stringByAppendingString:@"\nkernel void rt(device const float *d [[buffer(0)]], device const uint *sel [[buffer(1)]], device uint *o [[buffer(2)]], "
                                                  "uint i [[thread_position_in_grid]]) { uint r = rec_pack(d[i], sel[i]); o[3 * i] = r; o[3 * i + 1] = as_type<uint>(rec_depth(r)); o[3 * i + 2] = rec_code(r); }\n"];
    id<MTLComputePipelineState> pso = src_pso(gp, src, @"rt");
    const float ds[4] = {1.0f, 0x1p-62f, 0.37109375f + 0x1p-20f, 0.0f};
    float d[20];
    uint32_t sel[20];
    for (int a = 0; a < 4; a++) for (int k = 0; k < 5; k++) { d[a * 5 + k] = ds[a]; sel[a * 5 + k] = (uint32_t)k; }
    id<MTLBuffer> bd = src_buf(gp, d, sizeof d), bs = src_buf(gp, sel, sizeof sel), bo = src_buf(gp, NULL, 20 * 12);
    src_run(gp, pso, @[ bd, bs, bo ], 20);
    const uint32_t *o = bo.contents;
    int bad = 0, ctl_wrong_one = 0;
    for (int i = 0; i < 20; i++) {
        uint32_t db; memcpy(&db, &d[i], 4);
        bad += o[3 * i + 1] != db || o[3 * i + 2] != sel[i] || (o[3 * i] >> 29) != sel[i];
        /* control: the third bit restored to zero */
        uint32_t low = o[3 * i] & 0x1FFFFFFFu, cb = low;
        if (d[i] == 1.0f && cb != db) ctl_wrong_one++;
    }
    CHECKC(bad == 0, "im2-record-range: %d round trips wrong", bad);
    CHECKC(ctl_wrong_one == 5, "im2-record-range control: restoring the third bit as zero must fail on the depth one (%d of 5)", ctl_wrong_one);
    CHECKC(o[3 * 5 + 1] != 0, "im2-record-range: the smallest depth came back as zero");
}

/* A depth for which fast::divide (as used by the scale) and the correctly rounded division give different scales, found on the device. */
static float find_fast_depth(float Zs) {
    Gpu *gp = &R.gpu;
    NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void pr(device const float *d [[buffer(0)]], device uint *o [[buffer(1)]], constant float &zs [[buffer(2)]], uint i [[thread_position_in_grid]]) {"
                     " float n = fmax(1.0f - d[i], 1e-7f); o[2 * i] = as_type<uint>(zs * fast::divide(1.0f, n)); o[2 * i + 1] = as_type<uint>(zs / n); }\n";
    id<MTLComputePipelineState> pso = src_pso(gp, src, @"pr");
    enum { N = 8192 };
    float *d = malloc(N * 4);
    for (int k = 0; k < N; k++) d[k] = 0.3f + k * 0x1p-14f;
    id<MTLBuffer> bd = src_buf(gp, d, N * 4), bo = src_buf(gp, NULL, N * 8), bz = src_buf(gp, &Zs, 4);
    src_run(gp, pso, @[ bd, bo, bz ], N);
    const uint32_t *o = bo.contents;
    float found = -1;
    for (int k = 0; k < N; k++) if (o[2 * k] != o[2 * k + 1]) { found = d[k]; break; }
    free(d);
    return found;
}

static void im2_key_once(void) {
    S18 *s = new_scene(W_, H_, W_, H_);
    float d = find_fast_depth(s->Zs);
    if (d < 0) { printf("  im2-key-once: no depth with different quotients on this device: case void\n"); free(s); return; }
    for (int i = 0; i < W_ * H_; i++) s->dC[i] = s->dP[i] = 0.5f;
    fill_block(s, 100, 115, 60, 75, d, 3, 0);
    int n = W_ * H_;
    O18 *g = new_out(n);
    run_engine(s, g);
    int unwritten = 0;
    for (int i = 0; i < n; i++) for (int k = 0; k < 2; k++) {
        const uint32_t *w = k ? g->ip : g->ic;
        unwritten += w[2 * i] != 0xFFFFFFFFu && w[2 * i + 1] == 0xFFFFFFFFu;
    }
#if defined(METALFG_CTL_KEY_RECOMPUTE)
    CHECKC(unwritten > 0, "im2-key-once control (second pass recomputes the key with the correctly rounded division): no cell left without a payload");
#else
    CHECKC(unwritten == 0, "im2-key-once: %d occupied cells without a payload (depth %.9g)", unwritten, d);
#endif
    free(g->ic); free(g->ip); free(g->wp); free(g->wc); free(g); free(s);
}

static void im2_prev_rim(void) {
    S18 *s = new_scene(W_, H_, W_, H_);
    for (int i = 0; i < W_ * H_; i++) s->dC[i] = s->dP[i] = 0.5f;               /* scale 8 */
    fill_block(s, 100, 115, 60, 75, 0.6f, 0, 0);                                /* scale 10: distance 2, half of th_zn = 4 */
    for (int y = 60; y <= 75; y++) for (int x = 100; x <= 115; x++) { s->mvCx[ID(x, y)] = -6; s->mvPx[ID(x, y)] = 6; }
    int ctl[] = {X_ZNPREV, X_NOCLOSE};
    judge("im2-prev-rim", s, ctl, 2, 0, 0);
    free(s);
}

static void im2_bg_key(void) {
    float zs[2] = {0.45f, 1.0e8f};                       /* scale 0.9 (just below one), and far above one */
    for (int v = 0; v < 2; v++) {
        S18 *s = new_scene(W_, H_, W_, H_);
        s->Zs = zs[v];
        for (int i = 0; i < W_ * H_; i++) s->dC[i] = s->dP[i] = 0.5f;
        int n = W_ * H_;
        O18 *g = new_out(n);
        run_engine(s, g);
        int empty = 0;
        for (int i = 0; i < n; i++) empty += g->ic[2 * i] == 0xFFFFFFFFu || g->ip[2 * i] == 0xFFFFFFFFu;
        CHECKC(empty == 0, "im2-bg-key (scale %g): %d empty cells", s->Zs * 2, empty);
        uint32_t *kc = malloc(n * 4), *pc = malloc(n * 4), *kp = malloc(n * 4), *pp = malloc(n * 4);
        float *wp = malloc(n * 4), *wc = malloc(n * 4);
        scatter_ref(s, 0, 0, kc, pc); scatter_ref(s, 1, 0, kp, pp);
        weights_ref(s, 0, kc, pc, kp, pp, wp, wc);
        for (int i = 0; i < n; i++) { g->wp[i] = wp[i]; g->wc[i] = wc[i]; }
        CHECKC(cmp(s, kc, pc, kp, pp, g, wp, wc, NULL) == 0, "im2-bg-key (scale %g): differs from the model", s->Zs * 2);
        /* control: every key at or above K is empty; the model's cells all hold K, so all of them become empty */
        for (int i = 0; i < n; i++) { kc[i] = kp[i] = pc[i] = pp[i] = 0xFFFFFFFFu; }
        CHECKC(cmp(s, kc, pc, kp, pp, g, wp, wc, NULL) > 0, "im2-bg-key control (keys at or above K empty) did not differ from the engine");
        free(kc); free(pc); free(kp); free(pp); free(wp); free(wc);
        free(g->ic); free(g->ip); free(g->wp); free(g->wc); free(g); free(s);
    }
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        R = plain_rig(g);
        CH = [[MetalFGChain alloc] initWithDevice:g.dev library:R.lib Wm:W_ Hm:H_];
        if (!CH) { printf("chain kernels missing\n"); return 2; }
#if defined(METALFG_CTL_KEY_RECOMPUTE)
        im2_key_once();                 /* the control build runs only the case it controls */
        printf(fails ? "g_interm (key-recompute control build): %d FAILURES\n" : "g_interm (key-recompute control build): ok\n", fails);
        return fails != 0;
#endif
        im2_lowbits(); im2_record_range(); im2_key_once(); im2_prev_rim(); im2_bg_key();
        cases();
    }
    printf(fails ? "g_interm: %d FAILURES\n" : "g_interm: ok\n", fails);
    return fails != 0;
}
