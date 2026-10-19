#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "extract.h"
#include "sha256.h"
#include "weights.h"
#include "metalfgconst.h"

static int fail(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return 1;
}

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t rd32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t rd64(const uint8_t *p) { return rd32(p) | (uint64_t)rd32(p + 4) << 32; }

/* ---------------- BLAKE2b (RFC 7693), unkeyed ---------------- */

static const uint64_t B2IV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                                 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
static const uint8_t B2S[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#define B2G(a, b, c, d, x, y) do { \
    v[a] = v[a] + v[b] + (x); v[d] = ROTR64(v[d] ^ v[a], 32); v[c] = v[c] + v[d]; v[b] = ROTR64(v[b] ^ v[c], 24); \
    v[a] = v[a] + v[b] + (y); v[d] = ROTR64(v[d] ^ v[a], 16); v[c] = v[c] + v[d]; v[b] = ROTR64(v[b] ^ v[c], 63); } while (0)

static void b2_compress(uint64_t h[8], const uint8_t blk[128], uint64_t t, int last) {
    uint64_t m[16], v[16];
    for (int i = 0; i < 16; i++) m[i] = rd64(blk + 8 * i);
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = B2IV[i]; }
    v[12] ^= t;
    if (last) v[14] = ~v[14];
    for (int r = 0; r < 12; r++) {
        const uint8_t *s = B2S[r];
        B2G(0, 4, 8, 12, m[s[0]], m[s[1]]); B2G(1, 5, 9, 13, m[s[2]], m[s[3]]);
        B2G(2, 6, 10, 14, m[s[4]], m[s[5]]); B2G(3, 7, 11, 15, m[s[6]], m[s[7]]);
        B2G(0, 5, 10, 15, m[s[8]], m[s[9]]); B2G(1, 6, 11, 12, m[s[10]], m[s[11]]);
        B2G(2, 7, 8, 13, m[s[12]], m[s[13]]); B2G(3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

void fgx_blake2b(uint8_t *out, size_t outlen, const void *in, size_t inlen) {
    uint64_t h[8];
    memcpy(h, B2IV, sizeof h);
    h[0] ^= 0x01010000ull ^ (uint64_t)outlen;
    const uint8_t *p = in;
    uint64_t t = 0;
    while (inlen > 128) {
        t += 128;
        b2_compress(h, p, t, 0);
        p += 128; inlen -= 128;
    }
    uint8_t last[128] = {0};
    memcpy(last, p, inlen);
    t += inlen;
    b2_compress(h, last, t, 1);
    for (size_t i = 0; i < outlen; i++) out[i] = (uint8_t)(h[i / 8] >> (8 * (i % 8)));
}

/* ---------------- LZ4 block ---------------- */

long fgx_lz4_decode(const uint8_t *in, size_t n, uint8_t *out, size_t cap) {
    size_t ip = 0, op = 0;
    while (ip < n) {
        uint8_t tok = in[ip++];
        size_t lit = tok >> 4;
        if (lit == 15) {
            uint8_t b;
            do { if (ip >= n) return -1; b = in[ip++]; lit += b; } while (b == 255);
        }
        if (ip + lit > n || op + lit > cap) return -1;
        memcpy(out + op, in + ip, lit);
        ip += lit; op += lit;
        if (ip >= n) break;                      /* the last item has literals only */
        if (ip + 2 > n) return -1;
        size_t dist = in[ip] | in[ip + 1] << 8;
        ip += 2;
        size_t ml = (tok & 15);
        if (ml == 15) {
            uint8_t b;
            do { if (ip >= n) return -1; b = in[ip++]; ml += b; } while (b == 255);
        }
        ml += 4;
        if (dist == 0 || dist > op || op + ml > cap) return -1;
        for (size_t k = 0; k < ml; k++) { out[op] = out[op - dist]; op++; }   /* overlapping matches copy forward byte by byte */
    }
    return (long)op;
}

/* ---------------- tiled forms ---------------- */

static int finite16(uint16_t v) { return (v & 0x7c00) != 0x7c00; }

int fgx_untile_net1(const uint8_t *blob, uint32_t O, uint32_t K, uint32_t I, uint16_t *w, uint16_t *b, char *err, size_t cap) {
    if (O % 8 || (K != 1 && K != 3) || (K * I) % 16) return fail(err, cap, "net1 blob shape %ux%ux%ux%u is not tileable", O, K, K, I);
    size_t Wb = 2ull * O * K * K * I;
    uint32_t steps = K * I / 16, frag = 0;
    for (uint32_t nT = 0; nT < O / 8; nT++)
        for (uint32_t ky = 0; ky < K; ky++)
            for (uint32_t kS = 0; kS < steps; kS++, frag++) {
                const uint8_t *f = blob + 256ull * frag;
                for (uint32_t L = 0; L < 32; L++)
                    for (uint32_t e = 0; e < 4; e++) {
                        uint32_t row = (e < 2 ? 0 : 8) + 2 * (L % 4) + (e & 1), n = L / 4;
                        uint16_t v = rd16(f + 2 * (4 * L + e));
                        if (!finite16(v)) return fail(err, cap, "net1 blob holds a non-finite element");
                        uint32_t kk = 16 * kS + row, kx = K == 1 ? 0 : kk / I, i = K == 1 ? kk : kk % I;
                        w[(((size_t)(8 * nT + n) * K + ky) * K + kx) * I + i] = v;
                    }
            }
    for (size_t i = Wb; i < 2 * Wb; i++)
        if (blob[i]) return fail(err, cap, "net1 blob: the region after the weights is not zero");
    const uint8_t *bt = blob + 2 * Wb;
    for (uint32_t j = 0; j < O / 8; j++) {
        const uint8_t *tile = bt + 256ull * j;
        for (uint32_t r = 1; r < 16; r++)
            if (memcmp(tile, tile + 16 * r, 16) != 0) return fail(err, cap, "inconsistent bias tile");
        for (uint32_t c = 0; c < 8; c++) {
            uint16_t v = rd16(tile + 2 * c);
            if (!finite16(v)) return fail(err, cap, "net1 blob holds a non-finite bias");
            b[8 * j + c] = v;
        }
    }
    return 0;
}

int fgx_untile_net2(const uint8_t *blob, uint32_t O, uint32_t K, uint32_t Ist, uint32_t Ireal, uint16_t *w, char *err, size_t cap) {
    if (O % 8 || (K != 1 && K != 3) || Ist % 16) return fail(err, cap, "net2 blob shape %ux%ux%ux%u is not tileable", O, K, K, Ist);
    uint32_t taps = K * K, cB = Ist / 16;
    size_t tile = 0;
    for (uint32_t nT = 0; nT < O / 8; nT++)
        for (uint32_t tap = 0; tap < taps; tap++)
            for (uint32_t c = 0; c < cB; c++, tile++) {
                const uint8_t *t = blob + 256 * tile;
                for (uint32_t h = 0; h < 2; h++)
                    for (uint32_t n = 0; n < 8; n++)
                        for (uint32_t k = 0; k < 8; k++) {
                            uint16_t v = rd16(t + 2 * (64 * h + 8 * n + k));
                            uint32_t i = 16 * c + 8 * h + k;
                            if (!finite16(v)) return fail(err, cap, "net2 blob holds a non-finite element");
                            if (i >= Ireal && v) return fail(err, cap, "net2 blob: a padded input channel is not zero");
                            w[(((size_t)(8 * nT + n) * taps) + tap) * Ist + i] = v;
                        }
            }
    return 0;
}

/* ---------------- locator ---------------- */

/* One record per parameter-file entry except gx and gy (fg.extractor_r2), in parameter-file order. */
#define NREC (METALFG_PARAM_COUNT - 2)
static const char *PARAM_NAMES[NREC] = {"hm", "hf", "hb", "mt", "mfa", "mfh", "th_ui", "th_uia", "leak", "th_mask", "b_steer", "sat_n2",
                                        "d_dyn", "d_sim", "th_cam", "sp_min", "sp_max", "th_zn"};
static const int PARAM_INT[NREC] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0};

int fgx_locator_parse(const uint8_t *d, size_t n, FgxLocator *loc, char *err, size_t cap) {
    memset(loc, 0, sizeof *loc);
    if (n >= 8 && memcmp(d, "FGLOCAT", 7) == 0 && d[7] != '3') return fail(err, cap, "locator: unsupported locator version");
    if (n < 104 || memcmp(d, "FGLOCAT3", 8) != 0) return fail(err, cap, "locator: bad magic or too short");
    loc->nblobs = rd32(d + 8);
    loc->nparams = rd32(d + 12);
    memcpy(loc->payload_sha, d + 16, 32);
    memcpy(loc->params_sha, d + 48, 32);
    loc->container_magic = rd32(d + 80);
    loc->text_kind = rd32(d + 84) & 0xffff;
    loc->block_mask = rd64(d + 88);
    loc->unsupported_mask = rd64(d + 96);
    if (loc->nblobs != 67 || loc->nparams != NREC) return fail(err, cap, "locator: expected 67 blob and %d parameter records", NREC);
    if (n != 104 + 40ull * loc->nblobs + 72ull * loc->nparams) return fail(err, cap, "locator: size does not match the record counts");
    loc->blobs = calloc(loc->nblobs, sizeof(FgxBlobRec));
    loc->params = calloc(loc->nparams, sizeof(FgxParamRec));
    const uint8_t *p = d + 104;
    for (uint32_t i = 0; i < loc->nblobs; i++, p += 40) {
        FgxBlobRec *b = &loc->blobs[i];
        memcpy(b->fp, p, 8);
        b->tensor_index = rd32(p + 8); b->kind = rd32(p + 12); b->O = rd32(p + 16); b->K = rd32(p + 20);
        b->I_stored = rd32(p + 24); b->I_real = rd32(p + 28); b->variant = rd32(p + 32); b->blob_bytes = rd32(p + 36);
        uint64_t want = b->kind == 0 ? 2ull * (2ull * b->O * b->K * b->K * b->I_stored) + 256ull * (b->O / 8)
                      : b->kind == 1 ? 2ull * b->O * b->K * b->K * b->I_stored : b->kind == 2 ? 2ull * b->O : ~0ull;
        if (b->kind > 2 || want != b->blob_bytes || (b->variant != 0 && b->variant != 2) || b->tensor_index >= METALFG_TENSOR_COUNT || b->I_real > b->I_stored) {
            fgx_locator_free(loc);
            return fail(err, cap, "locator: blob record %u is inconsistent", i);
        }
    }
    for (uint32_t i = 0; i < loc->nparams; i++, p += 72) {
        FgxParamRec *r = &loc->params[i];
        memcpy(r->name, p, 24);
        r->name[24] = 0;
        memcpy(r->fp, p + 24, 8);
        r->wlen = rd16(p + 32); r->disp = (int16_t)rd16(p + 34); r->type = p[36]; r->deriv = p[37]; r->second = rd16(p + 38);
        r->area = p[40]; r->indirect = p[41]; r->tail = p[42]; r->nmask = p[43]; r->target = rd32(p + 44);
        for (int k = 0; k < 3; k++) { r->mask[k][0] = rd16(p + 48 + 4 * k); r->mask[k][1] = rd16(p + 50 + 4 * k); }
        r->addend = (int32_t)rd32(p + 60);
        r->prefilter = rd32(p + 64);
        r->prefix_len = p[68];
        int bad = strcmp(r->name, PARAM_NAMES[i]) != 0 || r->wlen > 128 || r->wlen < 8 || r->type > 7 || r->deriv > 4 || r->area > 2 || r->nmask > 3;
        for (int k = 0; k < r->nmask && !bad; k++)
            if (r->mask[k][0] < 8 || (uint32_t)r->mask[k][0] + r->mask[k][1] > r->wlen) bad = 1;
        if (r->deriv == 3 && r->second >= loc->nparams) bad = 1;
        if (r->indirect && r->area != 1) bad = 1;
        if (bad) {
            char nm[25];
            memcpy(nm, r->name, 25);
            fgx_locator_free(loc);
            return fail(err, cap, "locator: parameter record %u (%s) is inconsistent", i, nm);
        }
    }
    return 0;
}

void fgx_locator_free(FgxLocator *loc) {
    free(loc->blobs);
    free(loc->params);
    memset(loc, 0, sizeof *loc);
}

/* ---------------- PE image ---------------- */

typedef struct { uint32_t va, vsize, raw, rawsize, chars; } Sec;
typedef struct { Sec s[96]; int n; } Pe;

#define SCN_DATA 0x40u
#define SCN_EXEC 0x20000000u
#define SCN_READ 0x40000000u

static int pe_parse(const uint8_t *d, size_t n, Pe *pe) {
    if (n < 64 || d[0] != 'M' || d[1] != 'Z') return 0;
    uint32_t pos = rd32(d + 60);
    if ((uint64_t)pos + 24 > n || memcmp(d + pos, "PE\0\0", 4) != 0) return 0;
    uint16_t ns = rd16(d + pos + 6), szopt = rd16(d + pos + 20);
    uint32_t opt = pos + 24;
    if ((uint64_t)opt + szopt > n || szopt < 2) return 0;
    uint16_t magic = rd16(d + opt);
    if (magic != 0x10b && magic != 0x20b) return 0;
    uint64_t sh = (uint64_t)opt + szopt;
    if (ns > 96 || sh + 40ull * ns > n) return 0;
    pe->n = ns;
    for (int i = 0; i < ns; i++) {
        const uint8_t *h = d + sh + 40ull * i;
        pe->s[i].vsize = rd32(h + 8); pe->s[i].va = rd32(h + 12); pe->s[i].rawsize = rd32(h + 16);
        pe->s[i].raw = rd32(h + 20); pe->s[i].chars = rd32(h + 36);
        if ((uint64_t)pe->s[i].raw + pe->s[i].rawsize > n) return 0;
    }
    return 1;
}

static int is_data(const Sec *s) { return (s->chars & SCN_DATA) && (s->chars & SCN_READ) && !(s->chars & SCN_EXEC); }
static int is_code(const Sec *s) { return (s->chars & SCN_EXEC) != 0; }

/* ---------------- container texts ---------------- */

typedef struct { uint8_t *data; size_t len; uint32_t target; } Text;
typedef struct { Text *t; int n, cap; } TextList;

static void text_add(TextList *l, uint8_t *data, size_t len, uint32_t target) {
    if (l->n == l->cap) { l->cap = l->cap ? 2 * l->cap : 8; l->t = realloc(l->t, l->cap * sizeof(Text)); }
    l->t[l->n++] = (Text){data, len, target};
}

static void unpack_containers(const uint8_t *lib, const Pe *pe, const FgxLocator *loc, TextList *tl, FgxReport *rep) {
    for (int si = 0; si < pe->n; si++) {
        const Sec *s = &pe->s[si];
        if (!is_data(s)) continue;
        const uint8_t *base = lib + s->raw;
        for (uint64_t off = 0; off + 16 <= s->rawsize; off += 8) {
            if (rd32(base + off) != loc->container_magic) continue;
            uint16_t h0 = rd16(base + off + 6);
            uint64_t total = rd64(base + off + 8);
            if (h0 < 16 || total > s->rawsize || (uint64_t)h0 + total > s->rawsize - off) continue;     /* invalid: skipped silently */
            rep->containers++;
            uint64_t end = off + h0 + total, pos = off + h0;
            while (pos + 64 <= end) {
                const uint8_t *e = base + pos;
                uint32_t he = rd32(e + 4);
                if (he == 0) break;
                uint32_t S = (uint32_t)rd64(e + 8), C = rd32(e + 16);
                uint32_t target = rd32(e + 28);
                uint64_t flags = rd64(e + 40), U = rd64(e + 56);
                uint32_t stored = S > C ? S : C;
                if (pos + he + stored > end) { rep->entries_skipped++; break; }
                if (rd16(e) == loc->text_kind && !(flags & loc->unsupported_mask)) {
                    const uint8_t *pay = e + he;
                    uint8_t *txt = NULL;
                    long len = -1;
                    if (flags & loc->block_mask) {
                        txt = malloc(U + 1);
                        len = U < (1ull << 30) ? fgx_lz4_decode(pay, C, txt, U) : -1;
                        if (len != (long)U) { free(txt); txt = NULL; len = -1; }
                    } else {
                        txt = malloc(S + 1);
                        memcpy(txt, pay, S);
                        len = S;
                    }
                    if (txt) {
                        while (len > 0 && txt[len - 1] == 0) len--;
                        text_add(tl, txt, (size_t)len, target);
                        rep->entries_unpacked++;
                    } else rep->entries_skipped++;
                }
                pos += he + stored;
            }
        }
    }
}

/* ---------------- parameter search ---------------- */

static int hexv(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

/* read the stored form at p (inside [lo, hi)); returns 1 and the value, or 0 */
static int read_value(const uint8_t *p, const uint8_t *hi, int type, int prefix, double *out) {
    size_t avail = (size_t)(hi - p);
    switch (type) {
    case 0: { if (avail < 4) return 0; uint32_t b = rd32(p); float f; memcpy(&f, &b, 4); *out = f; return 1; }
    case 1: { if (avail < 8) return 0; uint64_t b = rd64(p); double f; memcpy(&f, &b, 8); *out = f; return 1; }
    case 2: if (avail < 4) return 0; *out = (int32_t)rd32(p); return 1;
    case 3: if (avail < 2) return 0; *out = (int16_t)rd16(p); return 1;
    case 4: if (avail < 1) return 0; *out = (int8_t)p[0]; return 1;
    case 5: case 7: {
        int nd = type == 5 ? 8 : 16;
        if (avail < (size_t)prefix + nd) return 0;
        uint64_t b = 0;
        for (int i = 0; i < nd; i++) { int v = hexv(p[prefix + i]); if (v < 0) return 0; b = b << 4 | (uint64_t)v; }
        if (type == 5) { uint32_t b32 = (uint32_t)b; float f; memcpy(&f, &b32, 4); *out = f; }
        else { double f; memcpy(&f, &b, 8); *out = f; }
        return 1;
    }
    case 6: {
        if (avail < (size_t)prefix + 1) return 0;
        const uint8_t *q = p + prefix;
        int neg = 0;
        if (q < hi && *q == '-') { neg = 1; q++; }
        double v = 0;
        int nd = 0;
        while (q < hi && *q >= '0' && *q <= '9') { v = v * 10 + (*q - '0'); q++; nd++; }
        if (!nd) return 0;
        *out = neg ? -v : v;
        return 1;
    }
    }
    return 0;
}

typedef struct { double v; int have; int ambiguous; } Hit;

static void note_hit(Hit *h, double v) {
    if (!h->have) { h->have = 1; h->v = v; }
    else if (!(h->v == v)) h->ambiguous = 1;
}

/* test one candidate window at p inside [lo, hi) */
static void try_window(const FgxParamRec *r, const uint8_t *p, const uint8_t *lo, const uint8_t *hi, const uint8_t *file,
                       const Pe *pe, Hit *h) {
    if ((size_t)(hi - p) < r->wlen) return;
    uint8_t win[128];
    memcpy(win, p, r->wlen);
    for (int k = 0; k < r->nmask; k++) memset(win + r->mask[k][0], 0, r->mask[k][1]);
    uint8_t dg[8];
    fgx_blake2b(dg, 8, win, r->wlen);
    if (memcmp(dg, r->fp, 8) != 0) return;
    const uint8_t *vp = p + r->disp;
    if (vp < lo) return;
    double v;
    if (r->indirect) {
        if (vp + 4 > hi) return;
        int32_t rel = (int32_t)rd32(vp);
        /* virtual address of the byte following the reference field, plus the tail, plus the reference */
        const uint8_t *next = vp + 4;
        int si = -1;
        for (int i = 0; i < pe->n; i++)
            if (file + pe->s[i].raw <= next - 1 && next - 1 < file + pe->s[i].raw + pe->s[i].rawsize) si = i;
        if (si < 0) return;
        int64_t va = (int64_t)pe->s[si].va + (int64_t)(next - (file + pe->s[si].raw)) + r->tail + rel;
        const uint8_t *tp = NULL, *thi = NULL;
        for (int i = 0; i < pe->n; i++)
            if (is_data(&pe->s[i]) && va >= pe->s[i].va && va < (int64_t)pe->s[i].va + pe->s[i].rawsize) {
                tp = file + pe->s[i].raw + (va - pe->s[i].va);
                thi = file + pe->s[i].raw + pe->s[i].rawsize;
            }
        if (!tp) return;
        if (!read_value(tp, thi, r->type, r->prefix_len, &v)) return;
    } else {
        if (!read_value(vp, hi, r->type, r->prefix_len, &v)) return;
    }
    note_hit(h, v);
}

static double derive(const FgxLocator *loc, const double *raw, int i, int depth) {
    const FgxParamRec *r = &loc->params[i];
    double v = raw[i];
    switch (r->deriv) {
    case 1: return 1.0 / v;
    case 2: return -v;
    case 3: return v * (depth < 8 ? derive(loc, raw, r->second, depth + 1) : NAN);
    case 4: return v + r->addend;
    }
    return v;
}

/* ---------------- main flow ---------------- */

static void names_missing(char *err, size_t cap, const char *what, const char *list) {
    fail(err, cap, "%s not found:%s", what, list);
}

int fgx_extract(const uint8_t *lib, size_t libsize, const FgxLocator *loc, uint8_t **weights, size_t *wsize, uint8_t **params,
                size_t *psize, FgxReport *rep, char *err, size_t cap) {
    memset(rep, 0, sizeof *rep);
    *weights = *params = NULL;
    Pe pe;
    if (!pe_parse(lib, libsize, &pe)) { fail(err, cap, "not a PE image"); return FGX_NOT_PE; }
    metalfg_sha256(lib, libsize, rep->lib_sha);
    rep->blobs_expected = (int)loc->nblobs;
    rep->params_expected = (int)loc->nparams;

    /* the locator's shapes must agree with the weight file table before anything is written into a tensor */
    const MetalFGTensorInfo *T = metalfg_weight_table();
    for (uint32_t i = 0; i < loc->nblobs; i++) {
        const FgxBlobRec *b = &loc->blobs[i];
        uint64_t welems = (uint64_t)b->O * b->K * b->K * b->I_stored;
        int ok = b->kind == 2 ? metalfg_tensor_elems(&T[b->tensor_index]) == b->O
                              : metalfg_tensor_elems(&T[b->tensor_index]) == welems && T[b->tensor_index].layout == 0;
        if (b->kind == 0) ok = ok && b->tensor_index + 1 < METALFG_TENSOR_COUNT && metalfg_tensor_elems(&T[b->tensor_index + 1]) == b->O;
        if (!ok) { fail(err, cap, "locator: the shape of blob %u does not match tensor %s", i, T[b->tensor_index].name); return FGX_NOT_FOUND; }
    }

    /* ---- blobs ---- */
    uint8_t **blobpos = calloc(loc->nblobs, sizeof(uint8_t *));
    uint16_t *tensor[METALFG_TENSOR_COUNT] = {0};
    int found = 0, rc = FGX_OK;
    for (int pass = 0; pass < 2 && found < (int)loc->nblobs; pass++) {
        size_t step = pass == 0 ? 16 : 4;
        rep->scan_step = (int)step;
        for (int si = 0; si < pe.n && found < (int)loc->nblobs; si++) {
            if (!is_data(&pe.s[si])) continue;
            const uint8_t *base = lib + pe.s[si].raw;
            size_t sz = pe.s[si].rawsize;
            for (size_t off = 0; off + 64 <= sz && found < (int)loc->nblobs; off += step) {
                uint8_t dg[8];
                int any = 0;
                for (uint32_t i = 0; i < loc->nblobs; i++) if (!blobpos[i]) { any = 1; break; }
                if (!any) break;
                fgx_blake2b(dg, 8, base + off, 64);
                for (uint32_t i = 0; i < loc->nblobs; i++) {
                    const FgxBlobRec *b = &loc->blobs[i];
                    if (blobpos[i] || memcmp(dg, b->fp, 8) != 0) continue;
                    if ((uint64_t)(base + off - lib) + b->blob_bytes > libsize) continue;
                    /* the structural signature of the kind must hold (decoys with a right fingerprint are skipped) */
                    const MetalFGTensorInfo *wt = &T[b->tensor_index];
                    char e2[200];
                    if (b->kind == 0) {
                        uint16_t *w = malloc(2ull * b->O * b->K * b->K * b->I_stored), *bb = malloc(2ull * b->O);
                        int ok = fgx_untile_net1(base + off, b->O, b->K, b->I_stored, w, bb, e2, sizeof e2) == 0;
                        free(w); free(bb);
                        if (!ok) continue;
                    } else if (b->kind == 1 && b->variant == 0) {
                        uint16_t *w = malloc(2ull * b->O * b->K * b->K * b->I_stored);
                        int ok = fgx_untile_net2(base + off, b->O, b->K, b->I_stored, b->I_real, w, e2, sizeof e2) == 0;
                        free(w);
                        if (!ok) continue;
                    } else {
                        int ok = 1;
                        for (uint32_t k = 0; k < b->blob_bytes / 2 && ok; k++) ok = finite16(rd16(base + off + 2 * k));
                        if (!ok) continue;
                    }
                    (void)wt;
                    blobpos[i] = (uint8_t *)(base + off);
                    found++;
                }
            }
        }
    }
    rep->blobs_found = found;
    if (found < (int)loc->nblobs) {
        char list[1500] = "";
        for (uint32_t i = 0; i < loc->nblobs && strlen(list) < 1300; i++)
            if (!blobpos[i]) { strcat(list, " "); strcat(list, T[loc->blobs[i].tensor_index].name); }
        names_missing(err, cap, "blob", list);
        rc = FGX_NOT_FOUND;
        goto done;
    }
    for (uint32_t i = 0; i < loc->nblobs && rc == FGX_OK; i++)          /* found blobs must not overlap */
        for (uint32_t j = i + 1; j < loc->nblobs; j++) {
            const uint8_t *a = blobpos[i], *b = blobpos[j];
            if (a < b + loc->blobs[j].blob_bytes && b < a + loc->blobs[i].blob_bytes) {
                fail(err, cap, "blobs for %s and %s overlap", T[loc->blobs[i].tensor_index].name, T[loc->blobs[j].tensor_index].name);
                rc = FGX_NOT_FOUND;
                break;
            }
        }
    if (rc) goto done;
    for (int t = 0; t < METALFG_TENSOR_COUNT; t++) tensor[t] = calloc(metalfg_tensor_elems(&T[t]), 2);
    for (uint32_t i = 0; i < loc->nblobs && rc == FGX_OK; i++) {
        const FgxBlobRec *b = &loc->blobs[i];
        const MetalFGTensorInfo *wt = &T[b->tensor_index];
        char e2[200];
        if (b->kind == 0) {
            if (fgx_untile_net1(blobpos[i], b->O, b->K, b->I_stored, tensor[b->tensor_index], tensor[b->tensor_index + 1], e2, sizeof e2)) {
                fail(err, cap, "%s: %s", wt->name, e2); rc = FGX_NOT_FOUND;
            }
        } else if (b->kind == 1 && b->variant == 0) {
            if (fgx_untile_net2(blobpos[i], b->O, b->K, b->I_stored, b->I_real, tensor[b->tensor_index], e2, sizeof e2)) {
                fail(err, cap, "%s: %s", wt->name, e2); rc = FGX_NOT_FOUND;
            }
        } else {                                               /* dense copy: variant 2 weights, or a bias */
            memcpy(tensor[b->tensor_index], blobpos[i], b->blob_bytes);
        }
        if (rc == FGX_OK && (uint64_t)b->blob_bytes > 0 && b->kind != 0 && wt->bytes != (b->kind == 2 ? b->blob_bytes : 2ull * b->O * b->K * b->K * b->I_stored)) {
            fail(err, cap, "%s: locator shape does not match the weight file table", wt->name); rc = FGX_NOT_FOUND;
        }
    }
    if (rc) goto done;

    /* ---- parameters ---- */
    {
        TextList tl = {0};
        unpack_containers(lib, &pe, loc, &tl, rep);
        double raw[NREC];
        Hit hits[NREC];
        memset(hits, 0, sizeof hits);
        /* the prefilter hash of the first eight bytes is computed once per position and compared with the records of the area */
        #define SCAN(AREA, LO, HI, STEP, TARGET)                                                                                      \
            for (const uint8_t *p = (LO); p + 8 <= (HI); p += (STEP)) {                                                            \
                uint32_t hh = metalfg_fnv1a32(p, 8);                                                                                  \
                for (uint32_t i = 0; i < loc->nparams; i++) {                                                                     \
                    const FgxParamRec *r = &loc->params[i];                                                                       \
                    if (r->area != (AREA) || r->prefilter != hh || ((AREA) == 2 && r->target && r->target != (TARGET))) continue;       \
                    try_window(r, p, (LO), (HI), lib, &pe, &hits[i]);                                                              \
                }                                                                                                                 \
            }
        for (int k = 0; k < tl.n; k++) {
            const uint8_t *lo = tl.t[k].data, *hi = lo + tl.t[k].len;
            SCAN(2, lo, hi, 1, tl.t[k].target)
        }
        for (int si = 0; si < pe.n; si++) {
            const Sec *s = &pe.s[si];
            const uint8_t *lo = lib + s->raw, *hi = lo + s->rawsize;
            if (is_data(s)) SCAN(0, lo, hi, 4, 0)
            if (is_code(s)) SCAN(1, lo, hi, 1, 0)
        }
        for (uint32_t i = 0; i < loc->nparams && rc == FGX_OK; i++) {
            const Hit h = hits[i];
            if (!h.have) { fail(err, cap, "parameter %s not found", loc->params[i].name); rc = FGX_NOT_FOUND; break; }
            if (h.ambiguous) { fail(err, cap, "ambiguous parameter %s", loc->params[i].name); rc = FGX_NOT_FOUND; break; }
            raw[i] = h.v;
            rep->params_found++;
        }
        for (int k = 0; k < tl.n; k++) free(tl.t[k].data);
        free(tl.t);
        if (rc) goto done;
        MetalFGParams P;
        memset(&P, 0, sizeof P);
        float *fp[NREC] = {0};
        int32_t *ip[NREC] = {0};
        ip[0] = &P.hm; ip[1] = &P.hf; ip[2] = &P.hb;
        fp[3] = &P.mt; fp[4] = &P.mfa; fp[5] = &P.mfh; fp[6] = &P.th_ui; fp[7] = &P.th_uia; fp[8] = &P.leak; fp[9] = &P.th_mask;
        fp[10] = &P.b_steer; fp[11] = &P.sat_n2; ip[12] = &P.d_dyn; ip[13] = &P.d_sim; fp[14] = &P.th_cam; ip[15] = &P.sp_min;
        ip[16] = &P.sp_max; fp[17] = &P.th_zn;
        P.gx = METALFG_GX; P.gy = METALFG_GY;       /* structural constants (fg.numerics_r3): written by the extractor, not found */
        for (int i = 0; i < NREC && rc == FGX_OK; i++) {
            double v = derive(loc, raw, i, 0);
            if (!isfinite(v)) { fail(err, cap, "parameter %s is not finite", PARAM_NAMES[i]); rc = FGX_NOT_FOUND; break; }
            if (PARAM_INT[i]) {
                if (v != floor(v) || v < -2147483648.0 || v > 2147483647.0) { fail(err, cap, "parameter %s is not an integer in range", PARAM_NAMES[i]); rc = FGX_NOT_FOUND; break; }
                *ip[i] = (int32_t)v;
            } else {
                float f = (float)v;
                if (!isfinite(f)) { fail(err, cap, "parameter %s is not finite as fp32", PARAM_NAMES[i]); rc = FGX_NOT_FOUND; break; }
                *fp[i] = f;
            }
        }
        if (rc) goto done;
        *params = metalfg_params_image(&P, psize);
    }

    /* ---- validation ---- */
    *weights = metalfg_weights_image((const uint16_t *const *)tensor, wsize);
    metalfg_sha256(*weights + METALFG_WEIGHT_DATA_OFFSET, *wsize - METALFG_WEIGHT_DATA_OFFSET, rep->payload_sha);
    {
        uint8_t ps[32];
        metalfg_sha256(*params, *psize - 32, ps);
        if (memcmp(rep->payload_sha, loc->payload_sha, 32) != 0) { fail(err, cap, "weight payload checksum mismatch"); rc = FGX_CHECKSUM; }
        else if (memcmp(ps, loc->params_sha, 32) != 0) { fail(err, cap, "parameter file checksum mismatch"); rc = FGX_CHECKSUM; }
    }
done:
    free(blobpos);
    for (int t = 0; t < METALFG_TENSOR_COUNT; t++) free(tensor[t]);
    if (rc) { free(*weights); free(*params); *weights = *params = NULL; }
    return rc;
}

char *fgx_discover(const uint8_t *lib, size_t libsize, const FgxLocator *loc) {
    size_t cap = 1 << 16, len = 0;
    char *out = malloc(cap);
    out[0] = 0;
    Pe pe;
    if (!pe_parse(lib, libsize, &pe)) { snprintf(out, cap, "not a PE image\n"); return out; }
#define APP(...) do { if (len + 256 > cap) { cap *= 2; out = realloc(out, cap); } len += (size_t)snprintf(out + len, cap - len, __VA_ARGS__); } while (0)
    APP("candidate net1 blobs (file offset, bytes, shape):\n");
    for (uint32_t i = 0; i < loc->nblobs; i++) {
        const FgxBlobRec *b = &loc->blobs[i];
        if (b->kind != 0) continue;
        size_t Wb = 2ull * b->O * b->K * b->K * b->I_stored;
        for (int si = 0; si < pe.n; si++) {
            if (!is_data(&pe.s[si])) continue;
            const uint8_t *base = lib + pe.s[si].raw;
            for (size_t off = 0; off + b->blob_bytes <= pe.s[si].rawsize; off += 16) {
                int z = 1;
                for (size_t k = 0; k < 64 && z; k++) z = base[off + Wb + k] == 0;
                if (!z) continue;
                uint16_t *w = malloc(Wb), *bb = malloc(2ull * b->O);
                char e2[100];
                if (fgx_untile_net1(base + off, b->O, b->K, b->I_stored, w, bb, e2, sizeof e2) == 0)
                    APP("  %zu  %u  %ux%ux%ux%u\n", (size_t)(base + off - lib), b->blob_bytes, b->O, b->K, b->K, b->I_stored);
                free(w); free(bb);
            }
        }
    }
    APP("containers (file offset):\n");
    FgxReport rep;
    memset(&rep, 0, sizeof rep);
    TextList tl = {0};
    unpack_containers(lib, &pe, loc, &tl, &rep);
    APP("  containers %d, entries unpacked %d, skipped %d\n", rep.containers, rep.entries_unpacked, rep.entries_skipped);
    for (int k = 0; k < tl.n; k++) { APP("  text %d: %zu bytes, target %u\n", k, tl.t[k].len, tl.t[k].target); free(tl.t[k].data); }
    free(tl.t);
    return out;
}
