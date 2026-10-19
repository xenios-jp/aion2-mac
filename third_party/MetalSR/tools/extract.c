/* extractsr: setup-time extractor (sr.extractor). Reads the user's own copy of the upscaler library, finds the network
   tensors and the named model parameters by fingerprint, writes the weight file and the parameter file the engine loads.
   Usage: extractsr <library> <locator> (--out <dir> | --verify) [-v]
   Exit status: 0 success, 1 usage or I/O error, 2 not a PE image, 3 item not found or out of range, 4 checksum mismatch. */
#include <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../src/model.h"

/* ---- BLAKE2b (RFC 7693), unkeyed, variable digest length ---- */
static const uint64_t B2_IV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                                  0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
static const uint8_t B2_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
static void b2_compress(uint64_t h[8], const uint8_t block[128], uint64_t t, int last) {
    uint64_t m[16], v[16];
    for (int i = 0; i < 16; i++) { m[i] = 0; for (int b = 0; b < 8; b++) m[i] |= (uint64_t)block[8 * i + b] << (8 * b); }
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = B2_IV[i]; }
    v[12] ^= t; if (last) v[14] = ~v[14];
#define G(a, b, c, d, x, y) do { v[a] = v[a] + v[b] + (x); v[d] = ROTR64(v[d] ^ v[a], 32); v[c] = v[c] + v[d]; v[b] = ROTR64(v[b] ^ v[c], 24); \
    v[a] = v[a] + v[b] + (y); v[d] = ROTR64(v[d] ^ v[a], 16); v[c] = v[c] + v[d]; v[b] = ROTR64(v[b] ^ v[c], 63); } while (0)
    for (int r = 0; r < 12; r++) {
        const uint8_t *s = B2_SIGMA[r];
        G(0, 4, 8, 12, m[s[0]], m[s[1]]); G(1, 5, 9, 13, m[s[2]], m[s[3]]); G(2, 6, 10, 14, m[s[4]], m[s[5]]); G(3, 7, 11, 15, m[s[6]], m[s[7]]);
        G(0, 5, 10, 15, m[s[8]], m[s[9]]); G(1, 6, 11, 12, m[s[10]], m[s[11]]); G(2, 7, 8, 13, m[s[12]], m[s[13]]); G(3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}
void sr_blake2b(const uint8_t *in, size_t len, uint8_t *out, int outlen) {
    uint64_t h[8]; memcpy(h, B2_IV, sizeof h);
    h[0] ^= 0x01010000ull ^ (uint64_t)outlen;
    uint8_t block[128];
    size_t pos = 0;
    while (len - pos > 128) { memcpy(block, in + pos, 128); pos += 128; b2_compress(h, block, pos, 0); }
    memset(block, 0, 128); memcpy(block, in + pos, len - pos);
    b2_compress(h, block, len, 1);
    for (int i = 0; i < outlen; i++) out[i] = (uint8_t)(h[i / 8] >> (8 * (i % 8)));
}

/* ---- fp32 -> binary16 by truncation toward zero ---- */
uint16_t sr_trunc_f16(uint32_t b) {
    uint32_t sign = (b >> 31) << 15, e = (b >> 23) & 0xFF, m = b & 0x7FFFFF;
    if (e == 0xFF) {
        if (m == 0) return (uint16_t)(sign | 0x7C00);                    /* infinity */
        return (uint16_t)(sign | 0x7E00 | (m >> 13));                    /* quiet NaN keeping the top 10 payload bits */
    }
    if (e == 0) return (uint16_t)sign;                                   /* fp32 subnormal / zero: below the smallest binary16 subnormal */
    int ex = (int)e - 127;
    if (ex > 15) return (uint16_t)(sign | 0x7BFF);                       /* at or above the largest finite value */
    if (ex >= -14) return (uint16_t)(sign | ((uint32_t)(ex + 15) << 10) | (m >> 13));
    /* binary16 subnormal: value = (1.m) * 2^ex = k * 2^-24 */
    int shift = -14 - ex;                                                /* 1..10 more bits dropped */
    if (shift > 10) return (uint16_t)sign;
    uint32_t full = 0x800000 | m;                                        /* 24-bit significand */
    return (uint16_t)(sign | (full >> (13 + shift)));
}

/* ---- PE helpers ---- */
typedef struct { const uint8_t *d; size_t n; } Buf;
static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
typedef struct { uint32_t va, vsize, raw, rawsize; } Section;

static int check_pe(Buf f, Section *secs, int *nsec, uint32_t *rsrc_rva, int *machine_ok, int *pe32plus) {
    if (f.n < 64 || f.d[0] != 'M' || f.d[1] != 'Z') return 0;
    uint32_t pe = rd32(f.d + 60);
    if ((uint64_t)pe + 24 > f.n || memcmp(f.d + pe, "PE\0\0", 4)) return 0;
    const uint8_t *coff = f.d + pe + 4;
    uint16_t machine = rd16(coff), ns = rd16(coff + 2), osz = rd16(coff + 16);
    *machine_ok = machine == 0x8664;
    uint64_t opt = (uint64_t)pe + 24, sec = opt + osz;
    if (opt + osz > f.n || osz < 2 || sec + 40ull * ns > f.n) return 0;
    uint16_t magic = rd16(f.d + opt);
    if (magic != 0x10B && magic != 0x20B) return 0;
    *pe32plus = magic == 0x20B;
    *nsec = ns > 96 ? 96 : ns;
    for (int i = 0; i < *nsec; i++) {
        const uint8_t *s = f.d + sec + 40 * i;
        secs[i] = (Section){rd32(s + 12), rd32(s + 8), rd32(s + 20), rd32(s + 16)};
    }
    *rsrc_rva = 0;
    uint64_t dd = opt + (magic == 0x20B ? 112 : 96) + 8 * 2;      /* data directory 2 = resources */
    if (dd + 8 <= opt + osz) *rsrc_rva = rd32(f.d + dd);
    return 1;
}
static int rva_to_off(const Section *secs, int ns, uint32_t rva, size_t fsize, size_t *off) {
    for (int i = 0; i < ns; i++) if (rva >= secs[i].va && rva < secs[i].va + (secs[i].vsize > secs[i].rawsize ? secs[i].vsize : secs[i].rawsize)) {
        size_t o = (size_t)secs[i].raw + (rva - secs[i].va);
        if (o < fsize) { *off = o; return 1; }
    }
    return 0;
}
/* file version from the first version resource (type 16, first name, first language); 0 on failure */
static int file_version(Buf f, const Section *secs, int ns, uint32_t rsrc_rva, unsigned v[4]) {
    size_t base;
    if (!rsrc_rva || !rva_to_off(secs, ns, rsrc_rva, f.n, &base)) return 0;
    size_t dir = base;
    for (int level = 0; level < 3; level++) {
        if (dir + 16 > f.n) return 0;
        unsigned named = rd16(f.d + dir + 12), ids = rd16(f.d + dir + 14);
        const uint8_t *ent = f.d + dir + 16;
        size_t pick = 0; int found = 0;
        for (unsigned i = 0; i < named + ids && dir + 16 + 8 * (i + 1) <= f.n; i++) {
            uint32_t name = rd32(ent + 8 * i), off = rd32(ent + 8 * i + 4);
            if (level == 0 && (name != 16 || (name & 0x80000000u))) continue;
            pick = off; found = 1; break;
        }
        if (!found) return 0;
        if (level < 2) { if (!(pick & 0x80000000u)) return 0; dir = base + (pick & 0x7FFFFFFFu); }
        else {   /* data entry */
            size_t de = base + pick;
            if (de + 16 > f.n) return 0;
            size_t doff;
            if (!rva_to_off(secs, ns, rd32(f.d + de), f.n, &doff)) return 0;
            uint32_t dsz = rd32(f.d + de + 4);
            for (size_t i = doff; i + 16 <= f.n && i < doff + dsz; i++)
                if (rd32(f.d + i) == 0xFEEF04BD) { uint32_t ms = rd32(f.d + i + 8), ls = rd32(f.d + i + 12); v[0] = ms >> 16; v[1] = ms & 0xFFFF; v[2] = ls >> 16; v[3] = ls & 0xFFFF; return 1; }
            return 0;
        }
    }
    return 0;
}

/* ---- locator ---- */
typedef struct { uint8_t fp[8]; uint32_t O, I, K, layout; } TensorRec;
typedef struct { char name[25]; uint8_t fp[8]; uint16_t wlen; int16_t disp; uint8_t type, deriv; uint16_t second; } ParamRec;
typedef struct { TensorRec t[SR_NUM_TENSORS]; ParamRec p[16]; uint8_t sha_w[32], sha_p[32]; } Locator;

static int load_locator(const char *path, Locator *L, char *why) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(why, 200, "cannot open locator %s", path); return 0; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(n);
    if (fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); snprintf(why, 200, "cannot read locator"); return 0; }
    fclose(f);
    size_t need = 16 + 24 * SR_NUM_TENSORS + 48 * 16 + 64;
    if ((size_t)n != need || memcmp(b, "SRLOCAT1", 8) || rd32(b + 8) != SR_NUM_TENSORS || rd32(b + 12) != 16) { free(b); snprintf(why, 200, "locator has the wrong format or size"); return 0; }
    const uint8_t *p = b + 16;
    for (int i = 0; i < SR_NUM_TENSORS; i++, p += 24) { memcpy(L->t[i].fp, p, 8); L->t[i].O = rd32(p + 8); L->t[i].I = rd32(p + 12); L->t[i].K = rd32(p + 16); L->t[i].layout = rd32(p + 20); }
    for (int i = 0; i < 16; i++, p += 48) {
        memcpy(L->p[i].name, p, 24); L->p[i].name[24] = 0; memcpy(L->p[i].fp, p + 24, 8);
        L->p[i].wlen = rd16(p + 32); L->p[i].disp = (int16_t)rd16(p + 34); L->p[i].type = p[36]; L->p[i].deriv = p[37]; L->p[i].second = rd16(p + 38);
    }
    memcpy(L->sha_w, p, 32); memcpy(L->sha_p, p + 32, 32);
    free(b);
    return 1;
}

/* ---- parameter derivations (sr.extractor_r3) ---- */
#ifdef EXTRACT_TEST_FAULTS
/* test-only fault injection for the controls of params-derive: 1 odds in double rounded once, 2 odds treated as none,
   3 read value rounded to a short decimal before widening */
int sr_derive_fault = 0;
#define DERIVE_FAULT sr_derive_fault
#else
#define DERIVE_FAULT 0
#endif

/* The stored value in its stored type, exactly widened to a double. */
static double read_stored(const uint8_t *s, int type) {
    if (type == 0) {
        float x; memcpy(&x, s, 4);
        if (DERIVE_FAULT == 3) { char b[40]; snprintf(b, sizeof b, "%.6g", (double)x); return strtod(b, NULL); }
        return (double)x;
    }
    if (type == 1) { double x; memcpy(&x, s, 8); return x; }
    int32_t x; memcpy(&x, s, 4); return (double)x;
}

/* Value of parameter record r (index i) from the read values raw[0..15]; the whole derivation table lives here. */
static double derive_value(const ParamRec *r, const double raw[16], int i) {
    double x = raw[i];
    switch (r->deriv) {
    case 1: return 1.0 / x;
    case 2: return -x;
    case 3: return x * raw[r->second];
    case 4: {
        if (DERIVE_FAULT == 1) return x / (1.0 - x);
        if (DERIVE_FAULT == 2) return x;
        volatile float fx = (float)x;          /* exact: the stored type is fp32 */
        volatile float d = 1.0f - fx;          /* first rounding, single precision */
        volatile float q = fx / d;             /* second rounding, single precision */
        return (double)q;
    }
    default: return x;
    }
}

/* Validity of the derivation fields: index of the first invalid record (reason in *why), or -1. */
static int check_derivations(const Locator *L, const char **why) {
    for (int i = 0; i < 16; i++) {
        const ParamRec *r = &L->p[i];
        if (r->deriv > 4) { *why = "unknown derivation code"; return i; }
        if (r->deriv == 3 && r->second >= 16) { *why = "second index not below the parameter count"; return i; }
        if (r->deriv == 4 && r->type != 0) { *why = "derivation 4 needs a stored fp32"; return i; }
    }
    return -1;
}

/* output type of each entry (sr.extractor_r2) */
static const int kTypeOf[16] = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};

/* ---- scan ---- */
typedef struct { uint8_t fp[8]; int len; int64_t pos; } Item;   /* tensors: len 64; parameters: window length */

static int scan(Buf f, Item *items, int n, int step) {
    int missing = 0;
    for (int i = 0; i < n; i++) if (items[i].pos < 0) missing++;
    int lens[64], nl = 0;
    for (int i = 0; i < n; i++) { int k = 0; while (k < nl && lens[k] != items[i].len) k++; if (k == nl && nl < 64) lens[nl++] = items[i].len; }
    for (size_t p = 0; missing && p + 64 <= f.n; p += step) {
        for (int k = 0; k < nl && missing; k++) {
            if (p + lens[k] > f.n) continue;
            uint8_t dg[8];
            sr_blake2b(f.d + p, lens[k], dg, 8);
            for (int i = 0; i < n; i++) if (items[i].pos < 0 && items[i].len == lens[k] && !memcmp(items[i].fp, dg, 8)) { items[i].pos = (int64_t)p; missing--; }
        }
    }
    return missing;
}

static uint16_t f16_of(float v) { uint32_t b; memcpy(&b, &v, 4); return sr_trunc_f16(b); }

static void hex(const uint8_t *d, int n, char *out) { for (int i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", d[i]); }

static int write_atomic(const char *dir, const char *name, const uint8_t *data, size_t n) {
    char tmp[1100], fin[1100];
    snprintf(fin, sizeof fin, "%s/%s", dir, name);
    snprintf(tmp, sizeof tmp, "%s/.%s.tmp%d", dir, name, getpid());
    FILE *f = fopen(tmp, "wb");
    if (!f) return 0;
    int ok = fwrite(data, 1, n, f) == n && fflush(f) == 0 && fsync(fileno(f)) == 0;
    ok = (fclose(f) == 0) && ok;
    if (ok) ok = chmod(tmp, 0644) == 0 && rename(tmp, fin) == 0;
    if (!ok) unlink(tmp);
    return ok;
}

static const char *kNames[16] = {"r_wide", "r_fast", "phi_lo", "phi_hi", "kappa_mv", "sigma_min", "sigma_span", "rho_max",
                                 "q_a", "q_b", "q_clip", "g_beta", "alpha_0", "key_ae", "eps_luma", "n_luma_min"};

int sr_extract(const char *libpath, const char *locpath, const char *outdir, int verbose, FILE *rep) {
    char why[256];
    FILE *lf = fopen(libpath, "rb");
    if (!lf) { fprintf(stderr, "cannot open %s: %s\n", libpath, strerror(errno)); return 1; }
    fseek(lf, 0, SEEK_END); long ln = ftell(lf); fseek(lf, 0, SEEK_SET);
    uint8_t *lib = malloc(ln > 0 ? ln : 1);
    if (!lib || fread(lib, 1, ln, lf) != (size_t)ln) { fclose(lf); fprintf(stderr, "cannot read %s\n", libpath); free(lib); return 1; }
    fclose(lf);
    Buf f = {lib, (size_t)ln};
    Locator L;
    if (!load_locator(locpath, &L, why)) { fprintf(stderr, "%s\n", why); free(lib); return 1; }

    { const char *dwhy = ""; int bad = check_derivations(&L, &dwhy);
      if (bad >= 0) { fprintf(stderr, "invalid locator: parameter record %d (%s): %s\n", bad, kNames[bad], dwhy); free(lib); return 1; } }

    Section secs[96]; int ns = 0, machine_ok = 0, pe32p = 0; uint32_t rsrc = 0;
    if (!check_pe(f, secs, &ns, &rsrc, &machine_ok, &pe32p)) { fprintf(stderr, "not a PE image\n"); free(lib); return 2; }
    if (!machine_ok) fprintf(stderr, "warning: machine type is not x86-64\n");
    uint8_t sha[32]; CC_SHA256(lib, (CC_LONG)ln, sha);
    char hx[80]; hex(sha, 32, hx);
    unsigned ver[4] = {0, 0, 0, 0};
    int hv = file_version(f, secs, ns, rsrc, ver);
    fprintf(rep, "library=%s\nlibrary_size=%ld\nlibrary_sha256=%s\n", libpath, ln, hx);
    if (hv) fprintf(rep, "library_version=%u.%u.%u.%u\n", ver[0], ver[1], ver[2], ver[3]); else fprintf(rep, "library_version=unknown\n");

    Item items[SR_NUM_TENSORS + 16];
    for (int i = 0; i < SR_NUM_TENSORS; i++) { memcpy(items[i].fp, L.t[i].fp, 8); items[i].len = 64; items[i].pos = -1; }
    for (int i = 0; i < 16; i++) { memcpy(items[SR_NUM_TENSORS + i].fp, L.p[i].fp, 8); items[SR_NUM_TENSORS + i].len = L.p[i].wlen; items[SR_NUM_TENSORS + i].pos = -1; }
    int step = 16, missing = scan(f, items, SR_NUM_TENSORS + 16, 16);
    if (missing) { step = 4; missing = scan(f, items, SR_NUM_TENSORS + 16, 4); }
    int found = SR_NUM_TENSORS + 16 - missing;
    if (missing) {
        if (found == 0) fprintf(stderr, "none of the expected items were found: this is not the supported library\n");
        else {
            for (int i = 0; i < SR_NUM_TENSORS; i++) if (items[i].pos < 0) fprintf(stderr, "missing: tensor %d (%s %s)\n", i, sr_tensor_shapes[i].layer, sr_tensor_shapes[i].is_bias ? "bias" : "weight");
            for (int i = 0; i < 16; i++) if (items[SR_NUM_TENSORS + i].pos < 0) fprintf(stderr, "missing: parameter %s\n", kNames[i]);
        }
        int tfound = 0;   /* tensors only: missing parameters do not change it */
        for (int i = 0; i < SR_NUM_TENSORS; i++) if (items[i].pos >= 0) tfound++;
        fprintf(rep, "tensors_found=%d\nscan_step=%d\nstatus=not found\n", tfound, step);
        free(lib); return 3;
    }
    fprintf(rep, "tensors_found=%d\nscan_step=%d\n", SR_NUM_TENSORS, step);

    /* weight image */
    size_t off[SR_NUM_TENSORS]; size_t total = sr_tensor_offsets(off);
    uint8_t *img = calloc(1, total);
    for (int t = 0; t < SR_NUM_TENSORS; t++) {
        const TensorRec *r = &L.t[t];
        size_t cnt = r->K ? (size_t)r->O * r->I * r->K * r->K : r->O;
        if ((uint64_t)items[t].pos + 4 * cnt > f.n) { fprintf(stderr, "tensor %d (%s) runs past the end of the file\n", t, sr_tensor_shapes[t].layer); free(lib); free(img); return 3; }
        const uint8_t *src = lib + items[t].pos;
        uint16_t *dst = (uint16_t *)(img + off[t]);
        for (size_t i = 0; i < cnt; i++) {
            size_t s = i;
            if (r->layout == 1) {            /* canonical (o, ky, kx, i) from source ((o * I + i) * K + ky) * K + kx */
                size_t K = r->K, I = r->I, ii = i % I, kx = (i / I) % K, ky = (i / (I * K)) % K, o = i / (I * K * K);
                s = ((o * I + ii) * K + ky) * K + kx;
            } else if (r->layout == 2) {     /* last layer: canonical row o from source row 10 * floor((o mod 8) / 2) + 2 * floor(o / 8) + (o mod 2) */
                size_t per = r->K ? (size_t)r->I : 1, o = i / per, rest = i % per;
                size_t srow = 10 * ((o % 8) / 2) + 2 * (o / 8) + (o % 2);
                s = srow * per + rest;
            }
            dst[i] = f16_of(*(const float *)(src + 4 * s));
        }
    }
    /* parameters */
    SrParams P; memset(&P, 0, sizeof P);
    double val[16], raw[16];
    for (int i = 0; i < 16; i++) {
        const ParamRec *r = &L.p[i];
        int64_t at = items[SR_NUM_TENSORS + i].pos + r->disp;
        size_t sz = r->type == 1 ? 8 : 4;
        if (at < 0 || (uint64_t)at + sz > f.n) { fprintf(stderr, "parameter %s is out of range\n", kNames[i]); free(lib); free(img); return 3; }
        const uint8_t *s = lib + at;
        raw[i] = read_stored(s, r->type);
    }
    for (int i = 0; i < 16; i++) {
        val[i] = derive_value(&L.p[i], raw, i);
        int ok = isfinite(val[i]);
        if (ok && kTypeOf[i] == 0) ok = isfinite((double)(float)val[i]);
        if (!ok) { fprintf(stderr, "parameter %s is not finite\n", kNames[i]); free(lib); free(img); return 3; }
    }
    double *slots[15] = {&P.r_wide, &P.r_fast, &P.phi_lo, &P.phi_hi, &P.kappa_mv, &P.sigma_min, &P.sigma_span, &P.rho_max, &P.q_a, &P.q_b, &P.q_clip, &P.g_beta, &P.alpha_0, &P.key_ae, &P.eps_luma};
    for (int i = 0; i < 15; i++) *slots[i] = val[i];
    if (val[15] != floor(val[15]) || val[15] < 0 || val[15] > 2147483647.0) { fprintf(stderr, "parameter n_luma_min is not an integer in range\n"); free(lib); free(img); return 3; }
    P.n_luma_min = (int32_t)val[15];
    uint8_t pbytes[16 + 16 * 40 + 32];
    size_t pn = sr_write_params(&P, pbytes);
    uint8_t shaw[32], shap[32];
    CC_SHA256(img, (CC_LONG)total, shaw);
    CC_SHA256(pbytes, 16 + 16 * 40, shap);
    hex(shaw, 32, hx); fprintf(rep, "weight_sha256=%s\n", hx);
    hex(shap, 32, hx); fprintf(rep, "params_sha256=%s\n", hx);
    if (memcmp(shaw, L.sha_w, 32) || memcmp(shap, L.sha_p, 32)) {
        fprintf(stderr, "checksum mismatch (%s%s): nothing written\n", memcmp(shaw, L.sha_w, 32) ? "weights" : "", memcmp(shap, L.sha_p, 32) ? " parameters" : "");
        fprintf(rep, "status=checksum mismatch\n"); free(lib); free(img); return 4;
    }
    if (outdir) {
        if (!write_atomic(outdir, "metalsr.weights", img, total) || !write_atomic(outdir, "metalsr.weights.params", pbytes, pn)) {
            fprintf(stderr, "cannot write to %s: %s\n", outdir, strerror(errno)); free(lib); free(img); return 1;
        }
    }
    fprintf(rep, "status=ok\n");
    (void)verbose;
    free(lib); free(img);
    return 0;
}

#ifndef EXTRACT_NO_MAIN
int main(int argc, char **argv) {
    const char *lib = NULL, *loc = NULL, *out = NULL; int verify = 0, verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--verify")) verify = 1;
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!lib) lib = argv[i];
        else if (!loc) loc = argv[i];
        else { lib = NULL; break; }
    }
    if (!lib || !loc || (!out && !verify) || (out && verify)) {
        fprintf(stderr, "usage: %s <library> <locator> (--out <dir> | --verify) [-v]\n", argv[0]);
        return 1;
    }
    return sr_extract(lib, loc, verify ? NULL : out, verbose, stdout);
}
#endif
