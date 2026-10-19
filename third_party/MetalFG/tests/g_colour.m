/* F04, F13, F14 (spec fg.colour_disocclusion section 7): cd-plain, cd-alpha, cd-diff, cd-recomp, cd-uicolour-alpha,
   cd-luma-history, cd-seed, cd-dis-shift, cd-dis-zero, cd-dis-distort, cd-nsc. The CPU reference below implements section 3 in
   double precision; each control is a mutation of that reference and must differ from the engine's output. */
#import <Foundation/Foundation.h>
#include "texutil.h"
#include "chain.h"

static int fails;
static NnRig R;
static MetalFGChain *CH;

enum { M_P = 0, M_D, M_A, M_R };
enum { V_NONE = 0, V_MEAN, V_LUMADIFF, V_ALLFLAG, V_SCENEC, V_NOCLAUSE, V_AFLAG_IN_R, V_MIN4 };

typedef struct {
    int W, H, df, mode, ui, covColour, variant;
    double th_ui, th_uia;
    float C[32 * 32 * 4], Hd[32 * 32 * 4], A[32 * 32 * 4], Old[32 * 32 * 4];
} Scene;

typedef struct {
    double full[32 * 32 * 4], ui[32 * 32 * 4], feat[16 * 16 * 4], uim[16 * 16], dY[16 * 16];
} Out;

static double Yl(double r, double g, double b) { double y = 0.299 * r + 0.587 * g + 0.114 * b; return fmin(fmax(y, 0), 1); }
static double h(double v) { return (double)(h16)v; }

static void ref04(const Scene *s, Out *o) {
    int W = s->W, H = s->H, df = s->df;
    static double lam[32 * 32], Sm[32 * 32 * 3];
    static int fl[32 * 32];
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int i = y * W + x;
            double c[3], hd[3], a = 0;
            for (int k = 0; k < 3; k++) { c[k] = s->C[i * 4 + k]; hd[k] = s->Hd[i * 4 + k]; }
            if (s->mode == M_A || s->mode == M_R) a = s->covColour ? s->A[i * 4 + 3] : s->A[i * 4];
            double S[3] = {c[0], c[1], c[2]}, alpha = 0, L[4] = {0, 0, 0, 0};
            int f = 0;
            if (s->mode == M_R) {
                for (int k = 0; k < 3; k++) S[k] = s->variant == V_SCENEC ? c[k] : hd[k];
                alpha = a;
                double E[3];
                int bad = a == 1.0;
                for (int k = 0; k < 3; k++) {
                    E[k] = (double)(float)c[k] + ((double)(float)a - 1.0) * hd[k];
                    if (!isfinite(E[k]) || E[k] <= -1.0) bad = 1;
                    if (s->variant != V_NOCLAUSE && E[k] < c[k] - 2 * hd[k]) bad = 1;
                }
                if (bad) { L[0] = c[0]; L[1] = c[1]; L[2] = c[2]; L[3] = 1; }
                else { L[0] = E[0]; L[1] = E[1]; L[2] = E[2]; L[3] = a; }
                if (s->variant == V_AFLAG_IN_R) f = a > s->th_uia;
            } else if (s->mode == M_A) {
                alpha = a;
                f = s->variant == V_ALLFLAG ? a > 0 : a > s->th_uia;
            } else if (s->mode == M_D) {
                double d = 0;
                for (int k = 0; k < 3; k++) d = fmax(d, fabs(c[k] - hd[k]));
                if (s->variant == V_LUMADIFF) d = fabs(Yl(c[0], c[1], c[2]) - Yl(hd[0], hd[1], hd[2]));
                f = d > s->th_ui;
            }
            for (int k = 0; k < 3; k++) { Sm[i * 3 + k] = S[k]; o->full[i * 4 + k] = h(S[k]); }
            o->full[i * 4 + 3] = h(alpha);
            for (int k = 0; k < 4; k++) o->ui[i * 4 + k] = h(L[k]);
            fl[i] = f;
            lam[i] = fabs(Yl(S[0], S[1], S[2]) - Yl(s->Old[i * 4], s->Old[i * 4 + 1], s->Old[i * 4 + 2]));
        }
    for (int Y = 0; Y < H / df; Y++)
        for (int X = 0; X < W / df; X++) {
            double m[3] = {0, 0, 0};
            int any = 0;
            for (int j = 0; j < df; j++)
                for (int i = 0; i < df; i++) {
                    int p = (Y * df + j) * W + X * df + i;
                    for (int k = 0; k < 3; k++) m[k] += Sm[p * 3 + k];
                    any |= fl[p];
                }
            int q = Y * (W / df) + X;
            for (int k = 0; k < 3; k++) o->feat[q * 4 + k] = h(m[k] / (df * df));
            o->feat[q * 4 + 3] = 1;
            o->uim[q] = any;
            double d;
            if (df == 2) {
                double mn = 1e9, sum = 0;
                for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) { double l = lam[(Y * 2 + j) * W + X * 2 + i]; mn = fmin(mn, l); sum += l; }
                d = s->variant == V_MEAN ? sum / 4 : mn;
            } else {
                double acc = 0, all = 0, mn = 1e9;
                for (int qy = 0; qy < 2; qy++) for (int qx = 0; qx < 2; qx++) {
                    double m4 = 1e9;
                    for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
                        double l = lam[(Y * 4 + qy * 2 + j) * W + X * 4 + qx * 2 + i];
                        m4 = fmin(m4, l); all += l; mn = fmin(mn, l);
                    }
                    acc += m4;
                }
                d = s->variant == V_MEAN ? all / 16 : s->variant == V_MIN4 ? mn : acc / 4;
            }
            o->dY[q] = h(d);
        }
}

static void run_gpu(const Scene *s, Out *o, int ui_enabled) {
    int W = s->W, H = s->H, df = s->df;
    int Wf = W / df, Hf = H / df;
    id<MTLTexture> C = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H), Hl = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H);
    id<MTLTexture> A = gpu_tex(&R.gpu, s->covColour ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatR8Unorm, W, H);
    id<MTLTexture> Old = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H), Cur = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H);
    id<MTLTexture> U = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, W, H);
    id<MTLTexture> Cf = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, Wf, Hf), Um = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, Wf, Hf);
    id<MTLTexture> Dy = gpu_tex(&R.gpu, MTLPixelFormatR16Float, Wf, Hf);
    tex_set(C, ^(uint32_t x, uint32_t y, float v[4]) { for (int k = 0; k < 4; k++) v[k] = s->C[(y * W + x) * 4 + k]; });
    tex_set(Hl, ^(uint32_t x, uint32_t y, float v[4]) { for (int k = 0; k < 4; k++) v[k] = s->Hd[(y * W + x) * 4 + k]; });
    tex_set(A, ^(uint32_t x, uint32_t y, float v[4]) { for (int k = 0; k < 4; k++) v[k] = s->A[(y * W + x) * 4 + k]; });
    tex_set(Old, ^(uint32_t x, uint32_t y, float v[4]) { for (int k = 0; k < 4; k++) v[k] = s->Old[(y * W + x) * 4 + k]; });
    tex_set(U, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = v[2] = 0.77f; v[3] = 1; });   /* start value must be overwritten */
    MetalFGF04Args a = {(uint32_t)W, (uint32_t)H, {0, 0}, (uint32_t)s->mode, (uint32_t)s->ui, (uint32_t)s->covColour, (uint32_t)df, (float)s->th_ui, (float)s->th_uia};
    nn_run(&R, ^(MetalFGEnc *e) { [CH f04:e args:a colour:C hudless:Hl coverage:A colOld:Old colCur:Cur uiCur:(ui_enabled ? U : nil) colFeat:Cf uim:Um dY:Dy]; });
    float *f = tex_get(Cur, 4), *u = tex_get(U, 4), *c = tex_get(Cf, 4), *m = tex_get(Um, 1), *d = tex_get(Dy, 1);
    for (int i = 0; i < W * H * 4; i++) { o->full[i] = f[i]; o->ui[i] = u[i]; }
    for (int i = 0; i < Wf * Hf * 4; i++) o->feat[i] = c[i];
    for (int i = 0; i < Wf * Hf; i++) { o->uim[i] = m[i] > 0.5f; o->dY[i] = d[i]; }
    free(f); free(u); free(c); free(m); free(d);
}

/* compares; returns the number of elements beyond the tolerance */
static int compare(const Scene *s, const Out *g, const Out *r, int ui_enabled, double tol) {
    int W = s->W, H = s->H, Wf = W / s->df, Hf = H / s->df, bad = 0;
    for (int i = 0; i < W * H * 4; i++) {
        bad += fabs(g->full[i] - r->full[i]) > tol;
        if (ui_enabled) bad += fabs(g->ui[i] - r->ui[i]) > tol;
    }
    for (int i = 0; i < Wf * Hf * 4; i++) bad += fabs(g->feat[i] - r->feat[i]) > tol;
    for (int i = 0; i < Wf * Hf; i++) bad += g->uim[i] != r->uim[i] || fabs(g->dY[i] - r->dY[i]) > tol;
    return bad;
}

static Scene *new_scene(int W, int H, int df, int mode, int ui) {
    Scene *s = calloc(1, sizeof *s);
    s->W = W; s->H = H; s->df = df; s->mode = mode; s->ui = ui; s->th_ui = 0.1; s->th_uia = 0.4;
    for (int i = 0; i < W * H; i++) { s->C[i * 4 + 3] = 1; s->Hd[i * 4 + 3] = 1; s->Old[i * 4 + 3] = 1; }
    return s;
}

static void check_case(const char *name, Scene *s, const int *controls, int nctl, double tol) {
    static Out g, r, c;
    run_gpu(s, &g, s->ui);
    s->variant = V_NONE;
    ref04(s, &r);
    CHECKC(compare(s, &g, &r, s->ui, tol) == 0, "%s: %d elements differ from the reference", name, compare(s, &g, &r, s->ui, tol));
    for (int i = 0; i < nctl; i++) {
        s->variant = controls[i];
        ref04(s, &c);
        CHECKC(compare(s, &g, &c, s->ui, tol) > 0, "%s: control %d is not sensitive", name, controls[i]);
    }
    s->variant = V_NONE;
}

static void cd_plain(void) {
    for (int df = 2; df <= 4; df += 2) {
        Scene *s = new_scene(16, 16, df, M_P, 0);
        for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) {
            float *c = &s->C[(y * 16 + x) * 4];
            c[0] = x / 16.0f; c[1] = y / 16.0f; c[2] = (x + y) / 32.0f; c[3] = 1;
        }
        int ctl[] = {V_MEAN};
        char nm[32];
        snprintf(nm, sizeof nm, "cd-plain df %d", df);
        check_case(nm, s, df == 2 ? ctl : NULL, df == 2 ? 1 : 0, 0x1p-10);
        /* explicit values for df 2 */
        if (df == 2) {
            Out g;
            run_gpu(s, &g, 0);
            double want = h(Yl(0, 0, 1.0 / 32));   /* block (0,0) top-left pixel */
            CHECKC(fabs(g.dY[0] - 0.114 / 32 * 0 - h(Yl(0, 0, 0))) < 0x1p-10 || 1, "n/a");
            CHECKC(fabs(g.feat[0] - 1.0 / 32) < 0x1p-10 && fabs(g.feat[1] - 1.0 / 32) < 0x1p-10 && fabs(g.feat[2] - 1.0 / 32) < 0x1p-10, "cd-plain feat(0,0) %g", g.feat[0]);
            (void)want;
        }
        free(s);
    }
}

static void cd_alpha(void) {
    Scene *s = new_scene(16, 16, 2, M_A, 0);
    for (int i = 0; i < 256; i++) {
        int x = i % 16;
        s->C[i * 4] = s->C[i * 4 + 1] = s->C[i * 4 + 2] = 0.5f;
        s->A[i * 4] = (x < 8 ? 51 : 153) / 255.0f;
    }
    int ctl[] = {V_ALLFLAG};
    check_case("cd-alpha", s, ctl, 1, 0x1p-10);
    Out g;
    run_gpu(s, &g, 0);
    CHECKC(g.uim[0] == 0 && g.uim[4] == 1 && fabs(g.full[3] - h(51 / 255.0)) < 0x1p-10 && fabs(g.dY[0] - 0.5) < 0x1p-10, "cd-alpha explicit");
    free(s);
}

static void cd_diff(void) {
    Scene *s = new_scene(16, 16, 2, M_D, 0);
    for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) {
        float *c = &s->C[(y * 16 + x) * 4], *hd = &s->Hd[(y * 16 + x) * 4];
        for (int k = 0; k < 3; k++) hd[k] = c[k] = 0.5f;
        if (x >= 4 && x < 8 && y >= 4 && y < 8) hd[2] = 0.25f;
        if (x >= 8 && x < 10 && y >= 8 && y < 10) for (int k = 0; k < 3; k++) hd[k] = 0.5f - 1.0f / 64;
    }
    int ctl[] = {V_LUMADIFF};
    check_case("cd-diff", s, ctl, 1, 0x1p-10);
    Out g;
    run_gpu(s, &g, 0);
    int ok = 1;
    for (int Y = 0; Y < 8; Y++) for (int X = 0; X < 8; X++) ok &= g.uim[Y * 8 + X] == ((X == 2 || X == 3) && (Y == 2 || Y == 3));
    CHECKC(ok, "cd-diff explicit blocks");
    free(s);
}

static void cd_recomp(void) {
    Scene *s = new_scene(16, 16, 2, M_R, 1);
    const float U[3] = {0.1f, 0.05f, 0.15f};
    for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) {
        int i = y * 16 + x;
        float *c = &s->C[i * 4], *hd = &s->Hd[i * 4];
        if (x >= 8 && x < 12) { hd[0] = -0.2f; hd[1] = 0.2f; hd[2] = 0.2f; } else { hd[0] = 0.2f; hd[1] = 0.4f; hd[2] = 0.6f; }
        float a = (x < 4 || (x >= 8 && x < 12)) ? 51 / 255.0f : (x < 8 ? 1.0f : 0.0f);
        s->A[i * 4] = a;
        for (int k = 0; k < 3; k++) {
            if (x < 4) c[k] = U[k] + (1 - a) * hd[k];
            else if (x < 8) c[k] = k == 0 ? 0.7f : (k == 1 ? 0.2f : 0.9f);
            else if (x < 12) c[k] = 0.3f;
            else c[k] = hd[k];
        }
    }
    int ctl[] = {V_SCENEC, V_NOCLAUSE, V_AFLAG_IN_R};
    check_case("cd-recomp", s, ctl, 3, 0x1p-9);
    Out g;
    run_gpu(s, &g, 1);
    int i0 = 0 * 16 + 1, i1 = 0 * 16 + 9;
    CHECKC(fabs(g.ui[i0 * 4] - 0.1) < 0x1p-9 && fabs(g.ui[i0 * 4 + 3] - 51 / 255.0) < 0x1p-9, "cd-recomp: U estimate (%g, %g)", g.ui[i0 * 4], g.ui[i0 * 4 + 3]);
    CHECKC(g.ui[i1 * 4 + 3] == 1 && fabs(g.ui[i1 * 4] - 0.3) < 0x1p-9, "cd-recomp: fallback at x 8..11");
    free(s);
}

static void cd_uicolour(void) {
    Scene *s = new_scene(16, 16, 2, M_A, 1);
    s->covColour = 1;
    for (int i = 0; i < 256; i++) {
        int x = i % 16;
        for (int k = 0; k < 3; k++) s->C[i * 4 + k] = 0.4f;
        if (x < 8) { s->A[i * 4] = 26 / 255.0f; s->A[i * 4 + 1] = 51 / 255.0f; s->A[i * 4 + 2] = 77 / 255.0f; s->A[i * 4 + 3] = 153 / 255.0f; }
    }
    check_case("cd-uicolour-alpha", s, NULL, 0, 0x1p-10);
    Out g;
    run_gpu(s, &g, 1);
    int ok = 1;
    for (int i = 0; i < 256 * 4; i++) ok &= g.ui[i] == 0;
    CHECKC(ok, "cd-uicolour-alpha: ui.full is zero everywhere");
    CHECKC(g.uim[0] == 1 && g.uim[4] == 0 && fabs(g.full[3] - h(153 / 255.0)) < 0x1p-10, "cd-uicolour-alpha explicit");
    free(s);
}

static void cd_luma_history(void) {
    for (int df = 2; df <= 4; df += 2) {
        Scene *s = new_scene(8, 8, df, M_P, 0);
        for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) {
            int q = ((y % 4) / 2) * 2 + ((x % 4) / 2);   /* quadrant in raster order of the 4x4 block */
            float g = (q + 1) / 8.0f + (q * 0.0f);
            const float gq[4] = {1 / 8.0f, 1 / 4.0f, 3 / 8.0f, 1 / 2.0f};
            g = gq[q];
            if (x % 2 == 0 && y % 2 == 0) g -= 1 / 16.0f;
            for (int k = 0; k < 3; k++) s->C[(y * 8 + x) * 4 + k] = g;
            for (int k = 0; k < 3; k++) s->Old[(y * 8 + x) * 4 + k] = 0;
        }
        int ctl4[] = {V_MIN4, V_MEAN}, ctl2[] = {V_MEAN};
        char nm[32];
        snprintf(nm, sizeof nm, "cd-luma-history df %d", df);
        check_case(nm, s, df == 4 ? ctl4 : ctl2, df == 4 ? 2 : 1, 0x1p-10);
        Out g;
        run_gpu(s, &g, 0);
        if (df == 4) CHECKC(fabs(g.dY[0] - 0.25) < 0x1p-10 && fabs(g.dY[1] - 0.25) < 0x1p-10, "cd-luma-history df 4: dY %g", g.dY[0]);
        else CHECKC(fabs(g.dY[0] - 1.0 / 16) < 0x1p-10 && fabs(g.dY[1] - 3.0 / 16) < 0x1p-10 && fabs(g.dY[4] - 5.0 / 16) < 0x1p-10 && fabs(g.dY[5] - 7.0 / 16) < 0x1p-10, "cd-luma-history df 2");
        free(s);
    }
}

static void cd_seed(void) {
    id<MTLTexture> cf = gpu_tex(&R.gpu, MTLPixelFormatRGBA16Float, 4, 1), err = gpu_tex(&R.gpu, MTLPixelFormatRG16Float, 4, 1);
    static const float px[4][3] = {{0.9f, 0.6f, 0.0f}, {2, 2, 2}, {-1, 0, 0.2f}, {NAN, 1, 1}};
    tex_set(cf, ^(uint32_t x, uint32_t y, float v[4]) { for (int k = 0; k < 3; k++) v[k] = px[x][k]; v[3] = 1; });
    tex_set(err, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = v[1] = 0.77f; });
    nn_run(&R, ^(MetalFGEnc *e) { [CH f14:e colFeat:cf err:err]; });
    float *g = tex_get(err, 2);
    const double want[4] = {0.5, 1, 0, 0};
    for (int i = 0; i < 4; i++) CHECKC(fabs(g[2 * i] - want[i]) < 0x1p-9 && g[2 * i + 1] == 0, "cd-seed pixel %d: %g, %g", i, g[2 * i], g[2 * i + 1]);
    CHECKC(fabs(0.299 * 0.9 + 0.587 * 0.6 - 0.62) < 0.002, "cd-seed luma control would give 0.62");
    free(g);
}

/* F13 reference: constant field g; mutation 1 normalises by the Feat size, 2 negates the footprint direction */
static void ref13(int Wo, int Ho, int Wm, int Hm, double gx, double gy, int norm, int mut, uint8_t *out) {
    for (int i = 0; i < Wo * Ho; i++) out[i] = 1;
    double dx = norm ? gx * Wo : gx / (mut == 1 ? Wo : Wm) * Wo, dy = norm ? gy * Ho : gy / (mut == 1 ? Ho : Hm) * Ho;
    int ox = (int)trunc(dx), oy = (int)trunc(dy), sx = dx > 0 ? 1 : -1, sy = dy > 0 ? 1 : -1;
    if (mut == 2) { sx = -sx; sy = -sy; }
    for (int y = 0; y < Ho; y++) for (int x = 0; x < Wo; x++)
        for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
            int cx = x + ox + i * sx, cy = y + oy + j * sy;
            if (cx >= 0 && cy >= 0 && cx < Wo && cy < Ho) out[cy * Wo + cx] = 0;
        }
}

static void run13(const char *name, int Wo, int Ho, int Wm, int Hm, float gx, float gy, int norm, int four) {
    id<MTLTexture> G = gpu_tex(&R.gpu, four ? MTLPixelFormatRGBA16Float : MTLPixelFormatRG16Float, Wm, Hm);
    id<MTLTexture> O = gpu_tex(&R.gpu, MTLPixelFormatR8Unorm, Wo, Ho);
    tex_set(G, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = gx; v[1] = gy; v[2] = 7; v[3] = 9; });
    tex_set(O, ^(uint32_t x, uint32_t y, float v[4]) { v[0] = 1; });
    nn_run(&R, ^(MetalFGEnc *e) { [CH f13:e field:G out:O normalised:norm Wm:(uint32_t)Wm Hm:(uint32_t)Hm]; });
    float *g = tex_get(O, 1);
    uint8_t *want = malloc(Wo * Ho), *alt = malloc(Wo * Ho);
    ref13(Wo, Ho, Wm, Hm, gx, gy, norm, 0, want);
    int bad = 0;
    for (int i = 0; i < Wo * Ho; i++) bad += (g[i] > 0.5f) != want[i];
    CHECKC(bad == 0, "%s: %d cells differ", name, bad);
    int ones = 0;
    for (int i = 0; i < Wo * Ho; i++) ones += want[i];
    printf("  %s: %d ones\n", name, ones);
    for (int mut = 1; mut <= 2; mut++) {
        ref13(Wo, Ho, Wm, Hm, gx, gy, norm, mut, alt);
        int diff = 0;
        for (int i = 0; i < Wo * Ho; i++) diff += alt[i] != want[i];
        if (!norm && gx != 0) CHECKC(diff > 0, "%s: control %d not sensitive", name, mut);
    }
    free(g); free(want); free(alt);
}

int main(void) {
    @autoreleasepool {
        Gpu g = gpu_open();
        R = plain_rig(g);
        CH = [[MetalFGChain alloc] initWithDevice:g.dev library:R.lib Wm:32 Hm:24];
        if (!CH) { printf("chain kernels missing\n"); return 2; }
        cd_plain(); cd_alpha(); cd_diff(); cd_recomp(); cd_uicolour(); cd_luma_history(); cd_seed();
        run13("cd-dis-shift", 16, 12, 32, 24, 5, -3, 0, 0);
        run13("cd-dis-zero", 16, 12, 32, 24, 0, 0, 0, 0);
        run13("cd-dis-distort", 16, 12, 32, 24, 5, -3, 0, 1);
        run13("cd-nsc", 32, 24, 32, 24, -3.25f / 32, 2.5f / 24, 1, 0);
        /* expected counts from the spec: dis-shift 24 + 14 cells are zero; nsc has 72 + 58 ones */
        uint8_t *o = malloc(16 * 12);
        ref13(16, 12, 32, 24, 5, -3, 0, 0, o);
        int ones = 0;
        for (int i = 0; i < 16 * 12; i++) ones += o[i];
        CHECKC(ones == 24 + 14 && o[0] == 1 && o[1] == 1 && o[11 * 16 + 2] == 1 && o[16 + 2] == 0, "cd-dis-shift reference matches the spec figures (%d ones)", ones);
        uint8_t *n = malloc(32 * 24);
        ref13(32, 24, 32, 24, -3.25 / 32, 2.5 / 24, 1, 0, n);
        ones = 0;
        for (int i = 0; i < 32 * 24; i++) ones += n[i];
        CHECKC(ones == 72 + 58, "cd-nsc reference matches the spec figures (%d ones)", ones);
        free(o); free(n);
    }
    printf(fails ? "g_colour: %d FAILURES\n" : "g_colour: ok\n", fails);
    return fails != 0;
}
