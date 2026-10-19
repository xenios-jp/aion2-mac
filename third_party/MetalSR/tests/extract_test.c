/* sr.extractor acceptance: extract-selftest, extract-fp16 (all 2^32 inputs), extract-synthetic. */
#define EXTRACT_NO_MAIN
#define EXTRACT_TEST_FAULTS
#include "../tools/extract.c"

static int fails;
#define EXPECT(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static void hexs(const uint8_t *d, int n, char *o) { for (int i = 0; i < n; i++) sprintf(o + 2 * i, "%02x", d[i]); }

/* independent model of the truncation, written from the specification text */
static uint16_t model_trunc(uint32_t bits) {
    float x; memcpy(&x, &bits, 4);
    uint16_t sign = (uint16_t)((bits >> 31) << 15);
    if (x != x) return (uint16_t)(sign | 0x7E00 | ((bits & 0x7FFFFF) >> 13));
    float a = fabsf(x);
    if (isinf(a)) return (uint16_t)(sign | 0x7C00);
    if (a >= 65504.0f) return (uint16_t)(sign | 0x7BFF);
    /* largest binary16 magnitude not above |x| */
    _Float16 h = (_Float16)a;                     /* nearest */
    uint16_t hb; memcpy(&hb, &h, 2);
    if ((float)h > a) hb--;
    return (uint16_t)(sign | hb);
}

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }

static void put32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static void put16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

static int read_file(const char *path, uint8_t *dst, size_t n) { FILE *f = fopen(path, "rb"); if (!f) return 0; size_t g = fread(dst, 1, n + 1, f); fclose(f); return g == n; }

/* ---- synthetic library builder ---- */
enum { FILESZ = 6 * 1024 * 1024, DATA0 = 0x400 };
typedef struct {
    double stored[16]; uint8_t stype[16], deriv[16]; uint16_t second[16];   /* what the image holds and what the locator says */
    double expect[16];                                                      /* the harness's own result per entry */
} Spec;
typedef struct {
    uint8_t *pe, *want; size_t total, tpos[SR_NUM_TENSORS], pn;
    uint8_t loc[16 + 24 * SR_NUM_TENSORS + 48 * 16 + 64], wantp[16 + 16 * 40 + 32];
} Synth;

/* PE image with known random tensors and parameters at random 4-byte-aligned positions, its locator, and the canonical
   outputs built by the harness's own model */
static void synth_build(Synth *Y, const Spec *sp) {
    uint8_t *pe = calloc(1, FILESZ), *loc = Y->loc;
    pe[0] = 'M'; pe[1] = 'Z'; put32(pe + 60, 0x80);
    memcpy(pe + 0x80, "PE\0\0", 4);
    put16(pe + 0x84, 0x8664); put16(pe + 0x86, 1); put16(pe + 0x84 + 16, 240);
    put16(pe + 0x98, 0x20B);                                   /* optional header (PE32+) */
    uint8_t *sec = pe + 0x98 + 240;
    memcpy(sec, ".data", 5); put32(sec + 8, FILESZ - DATA0); put32(sec + 12, 0x1000); put32(sec + 16, FILESZ - DATA0); put32(sec + 20, DATA0);
    size_t off[SR_NUM_TENSORS]; size_t total = sr_tensor_offsets(off);
    uint8_t *want = calloc(1, total);
    size_t cursor = DATA0 + 1024;
    memset(loc, 0, sizeof Y->loc);
    memcpy(loc, "SRLOCAT1", 8); put32(loc + 8, SR_NUM_TENSORS); put32(loc + 12, 16);
    for (int t = 0; t < SR_NUM_TENSORS; t++) {
        const SrTensorShape *s = &sr_tensor_shapes[t];
        size_t cnt = s->is_bias ? s->o : (size_t)s->o * s->i * s->k * s->k;
        cursor += 4 * (rnd() % 64);                                 /* 4-byte aligned, usually not 16-byte aligned */
        Y->tpos[t] = cursor;
        float *src = (float *)(pe + cursor);
        for (size_t i = 0; i < cnt; i++) { uint32_t r = rnd(); float v = ((float)(r & 0xFFFF) - 32768.0f) / 8192.0f; if (r % 997 == 0) v *= 1e6f; if (r % 1013 == 0) v = 1e-9f; src[i] = v; }
        cursor += 4 * cnt + 8;
        uint32_t layout = (s->k == 3) ? 1 : ((t >= 32) ? 2 : 0);
        /* canonical image by an independent loop structure */
        uint16_t *dst = (uint16_t *)(want + off[t]);
        if (s->is_bias) {
            for (uint32_t o = 0; o < s->o; o++) { uint32_t so = layout == 2 ? 10 * ((o % 8) / 2) + 2 * (o / 8) + (o % 2) : o; uint32_t b; memcpy(&b, &src[so], 4); dst[o] = model_trunc(b); }
        } else if (layout == 1) {
            for (uint32_t o = 0; o < s->o; o++) for (uint32_t ky = 0; ky < 3; ky++) for (uint32_t kx = 0; kx < 3; kx++) for (uint32_t i = 0; i < s->i; i++) {
                uint32_t b; memcpy(&b, &src[(((size_t)o * s->i + i) * 3 + ky) * 3 + kx], 4);
                dst[(((size_t)o * 3 + ky) * 3 + kx) * s->i + i] = model_trunc(b);
            }
        } else {
            for (uint32_t o = 0; o < s->o; o++) for (uint32_t i = 0; i < s->i; i++) {
                uint32_t so = layout == 2 ? 10 * ((o % 8) / 2) + 2 * (o / 8) + (o % 2) : o; uint32_t b; memcpy(&b, &src[(size_t)so * s->i + i], 4);
                dst[(size_t)o * s->i + i] = model_trunc(b);
            }
        }
        uint8_t dg[8]; sr_blake2b(pe + cursor - 4 * cnt - 8, 64, dg, 8);
        uint8_t *r = loc + 16 + 24 * t;
        memcpy(r, dg, 8); put32(r + 8, s->o); put32(r + 12, s->is_bias ? 0 : s->i); put32(r + 16, s->is_bias ? 0 : s->k); put32(r + 20, layout);
    }
    /* parameters: 16-byte windows holding the stored value at displacement 4 */
    SrParams P; memset(&P, 0, sizeof P);
    for (int i = 0; i < 16; i++) {
        cursor = (cursor + 3) & ~(size_t)3; size_t at = cursor; cursor += 32;
        for (int b = 0; b < 16; b++) pe[at + b] = (uint8_t)rnd();
        if (sp->stype[i] == 0) { float x = (float)sp->stored[i]; memcpy(pe + at + 4, &x, 4); }
        else if (sp->stype[i] == 1) memcpy(pe + at + 4, &sp->stored[i], 8);
        else { int32_t x = (int32_t)sp->stored[i]; memcpy(pe + at + 4, &x, 4); }
        uint8_t dg[8]; sr_blake2b(pe + at, 16, dg, 8);
        uint8_t *r = loc + 16 + 24 * SR_NUM_TENSORS + 48 * i;
        strncpy((char *)r, kNames[i], 24); memcpy(r + 24, dg, 8); put16(r + 32, 16); put16(r + 34, 4);
        r[36] = sp->stype[i]; r[37] = sp->deriv[i]; put16(r + 38, sp->deriv[i] == 3 ? sp->second[i] : 65535);
    }
    double *slots[15] = {&P.r_wide, &P.r_fast, &P.phi_lo, &P.phi_hi, &P.kappa_mv, &P.sigma_min, &P.sigma_span, &P.rho_max, &P.q_a, &P.q_b, &P.q_clip, &P.g_beta, &P.alpha_0, &P.key_ae, &P.eps_luma};
    for (int i = 0; i < 15; i++) *slots[i] = sp->expect[i];
    P.n_luma_min = (int32_t)sp->expect[15];
    Y->pn = sr_write_params(&P, Y->wantp);
    uint8_t *tail = loc + 16 + 24 * SR_NUM_TENSORS + 48 * 16;
    CC_SHA256(want, (CC_LONG)total, tail); CC_SHA256(Y->wantp, 16 + 16 * 40, tail + 32);
    Y->pe = pe; Y->want = want; Y->total = total;
}

static int run_synth(const Synth *Y, const char *outdir) {
    FILE *f = fopen("build/x_lib.dll", "wb"); fwrite(Y->pe, 1, FILESZ, f); fclose(f);
    f = fopen("build/x_loc.bin", "wb"); fwrite(Y->loc, 1, sizeof Y->loc, f); fclose(f);
    FILE *rep = fopen("build/x_report.txt", "w");
    int rc = sr_extract("build/x_lib.dll", "build/x_loc.bin", outdir, 0, rep);
    fclose(rep);
    return rc;
}

/* the default spec: plain copies (derivation 0) of fp32 stored values */
static void default_spec(Spec *sp) {
    double pv[16] = {2.5, 1.8, 0.00024414062, 0.00048828125, 8.0, 0.3125, 0.6875, 0.875, 1.0, 1.75, 1.5, 1.0, 0.5, 0.1796875, 0.0001, 16};
    memset(sp, 0, sizeof *sp);
    for (int i = 0; i < 16; i++) { sp->stored[i] = pv[i]; sp->stype[i] = (i == 15) ? 2 : 0; sp->second[i] = 65535; sp->expect[i] = i == 15 ? 16 : (double)(float)pv[i]; }
}

/* the single-precision odds of the spec, two roundings, written with separate float temporaries */
static float model_odds(float x) { volatile float d = 1.0f - x; volatile float q = x / d; return q; }

/* a random fp32 in (0, 1) whose single-precision odds differ from double evaluation rounded once; 0 when none is found */
static float find_odds_case(void) {
    for (int n = 0; n < 1000000; n++) {
        uint32_t bits = ((uint32_t)(110 + rnd() % 17) << 23) | (rnd() & 0x7FFFFF);   /* [2^-17, 1) with a full significand */
        float x; memcpy(&x, &bits, 4);
        float single = model_odds(x), once = (float)((double)x / (1.0 - (double)x));
        if (single != once) return x;
    }
    return 0.0f;
}

static int report_tensors_found(void) {
    char line[200]; int tf = -1; FILE *rp = fopen("build/x_report.txt", "r");
    while (rp && fgets(line, sizeof line, rp)) sscanf(line, "tensors_found=%d", &tf);
    if (rp) fclose(rp);
    return tf;
}

static int params_equal(const char *dir, const Synth *Y) {
    char path[200]; uint8_t gotp[16 + 16 * 40 + 32];
    snprintf(path, sizeof path, "%s/metalsr.weights.params", dir);
    return read_file(path, gotp, Y->pn) && !memcmp(gotp, Y->wantp, Y->pn);
}

int main(void) {
    /* extract-selftest */
    {
        uint8_t d[64], s[32]; char hx[140];
        sr_blake2b((const uint8_t *)"abc", 3, d, 64); hexs(d, 64, hx);
        EXPECT(!strcmp(hx, "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"), "BLAKE2b-512(abc)");
        CC_SHA256("abc", 3, s); hexs(s, 32, hx);
        EXPECT(!strcmp(hx, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "SHA-256(abc)");
        uint8_t z[64] = {0}, d8[8]; sr_blake2b(z, 64, d8, 8); hexs(d8, 8, hx);
        printf("extract-selftest: BLAKE2b-8 of 64 zero bytes = %s (for an external cross-check)\n", hx);
    }
    /* extract-fp16 over every fp32 bit pattern */
    {
        uint64_t bad = 0;
        for (uint64_t b = 0; b <= 0xFFFFFFFFull; b++) {
            uint16_t got = sr_trunc_f16((uint32_t)b), want = model_trunc((uint32_t)b);
            if (got != want) { if (bad < 5) printf("  fp16 mismatch for %08llx: %04x vs %04x\n", (unsigned long long)b, got, want); bad++; }
        }
        printf("extract-fp16: %llu mismatches over 2^32 inputs\n", (unsigned long long)bad);
        EXPECT(bad == 0, "extract-fp16");
        /* control: round-to-nearest differs */
        int differs = 0;
        for (uint32_t b = 0x3F800000; b < 0x3F800000 + 0x800000; b += 0x1001) { float x; memcpy(&x, &b, 4); _Float16 h = (_Float16)x; uint16_t hb; memcpy(&hb, &h, 2); if (hb != sr_trunc_f16(b)) differs = 1; }
        EXPECT(differs, "control: nearest rounding must differ from truncation");
    }
    /* extract-synthetic */
    {
        Spec sp; default_spec(&sp);
        Synth Y; synth_build(&Y, &sp);
        mkdir("build/x_out", 0755);
        int rc = run_synth(&Y, "build/x_out");
        printf("extract-synthetic: exit %d\n", rc);
        EXPECT(rc == 0, "extractor exit status %d", rc);
        uint8_t *got = malloc(Y.total);
        EXPECT(read_file("build/x_out/metalsr.weights", got, Y.total) && !memcmp(got, Y.want, Y.total), "weight file equals the canonical image");
        EXPECT(params_equal("build/x_out", &Y), "parameter file equals the canonical image");
        EXPECT(report_tensors_found() == SR_NUM_TENSORS, "tensors_found=%d on a complete run", report_tensors_found());
        /* the engine's loaders accept the outputs */
        char err[300];
        uint8_t *w2 = sr_load_weights("build/x_out/metalsr.weights", err, sizeof err);
        SrParams P2; EXPECT(w2 && sr_load_params("build/x_out/metalsr.weights", &P2, err, sizeof err) == 0 && P2.n_luma_min == 16, "engine loads the extractor output (%s)", err);
        free(w2); free(got);
        /* verify-only mode writes nothing and succeeds */
        unlink("build/x_out/metalsr.weights");
        rc = run_synth(&Y, NULL);
        EXPECT(rc == 0 && access("build/x_out/metalsr.weights", F_OK) != 0, "verify-only run");
        /* a tensor moved to an unaligned position must make the run fail with exit 3 */
        { size_t t = 10, n2 = (size_t)sr_tensor_shapes[t].o * sr_tensor_shapes[t].i * 4;
          Synth Z = Y; Z.pe = malloc(FILESZ); memcpy(Z.pe, Y.pe, FILESZ);
          memmove(Z.pe + Y.tpos[t] + 1, Z.pe + Y.tpos[t], n2);
          rc = run_synth(&Z, "build/x_out2"); free(Z.pe);
          printf("  unaligned tensor: exit %d (expected 3)\n", rc);
          EXPECT(rc == 3, "unaligned tensor must give exit 3"); }
        /* corrupted checksum in the locator: exit 4, nothing written */
        { Synth Z = Y; Z.loc[sizeof Z.loc - 1] ^= 1;
          mkdir("build/x_out3", 0755);
          rc = run_synth(&Z, "build/x_out3");
          EXPECT(rc == 4 && access("build/x_out3/metalsr.weights", F_OK) != 0, "checksum mismatch must give exit 4 and write nothing (rc %d)", rc); }
        /* not a PE image */
        { FILE *f = fopen("build/x_notpe.bin", "wb"); fwrite("hello", 1, 5, f); fclose(f);
          FILE *rep = fopen("build/x_report.txt", "w"); rc = sr_extract("build/x_notpe.bin", "build/x_loc.bin", NULL, 0, rep); fclose(rep);
          EXPECT(rc == 2, "not a PE image must give exit 2 (rc %d)", rc); }
        /* tensors_found counts tensors only: with only a parameter missing it is still all of them */
        { Synth Z = Y; Z.loc[16 + 24 * SR_NUM_TENSORS + 48 * 3 + 24] ^= 0xFF;     /* fingerprint of parameter record 3 no longer matches */
          rc = run_synth(&Z, NULL);
          int tf = report_tensors_found();
          printf("  only a parameter missing: exit %d, tensors_found=%d\n", rc, tf);
          EXPECT(rc == 3 && tf == SR_NUM_TENSORS, "tensors_found=%d with only a parameter missing (rc %d)", tf, rc);
          Z.loc[16 + 24 * 5] ^= 0xFF;                                             /* and tensor 5 missing as well */
          rc = run_synth(&Z, NULL); tf = report_tensors_found();
          EXPECT(rc == 3 && tf == SR_NUM_TENSORS - 1, "tensors_found=%d with a tensor and a parameter missing (rc %d)", tf, rc); }
        free(Y.pe); free(Y.want);
    }
    /* params-derive: one record of each derivation code; params-codes: invalid locators */
    {
        float xo = find_odds_case();
        EXPECT(xo != 0.0f, "params-derive: no fp32 value in (0, 1) separates single from double evaluation");
        printf("params-derive: odds case x=%.9g single=%.9g double-once=%.9g\n", (double)xo, (double)model_odds(xo), (double)(float)((double)xo / (1.0 - (double)xo)));
        Spec sp; default_spec(&sp);
        /* 0 none: fp32 1.8 into an fp64 entry (the widened single, not the short decimal) */
        sp.stored[1] = 1.8; sp.expect[1] = (double)(float)1.8;
        /* 1 reciprocal into an fp64 entry: one double division */
        sp.stored[0] = 3.7; sp.deriv[0] = 1; sp.expect[0] = 1.0 / (double)(float)3.7;
        /* 2 negation, fp32 entry */
        sp.stored[2] = 0.3; sp.deriv[2] = 2; sp.expect[2] = (double)(float)(-(double)(float)0.3);
        /* 3 product with record 4 */
        sp.stored[3] = 0.7; sp.deriv[3] = 3; sp.second[3] = 4; sp.stored[4] = 5.3;
        sp.expect[4] = (double)(float)5.3; sp.expect[3] = (double)(float)((double)(float)0.7 * (double)(float)5.3);
        /* 0 none from a stored fp64 into an fp32 entry: round to nearest even */
        sp.stored[6] = 0.6875123456789; sp.stype[6] = 1; sp.expect[6] = (double)(float)0.6875123456789;
        /* 4 odds */
        sp.stored[13] = (double)xo; sp.deriv[13] = 4; sp.expect[13] = (double)model_odds(xo);
        Synth Y; synth_build(&Y, &sp);
        mkdir("build/x_outd", 0755);
        int rc = run_synth(&Y, "build/x_outd");
        EXPECT(rc == 0, "params-derive: exit %d", rc);
        EXPECT(params_equal("build/x_outd", &Y), "params-derive: parameter file equals the harness's file");
        /* controls: each wrong extractor must fail with a checksum mismatch and write nothing */
        const char *names[3] = {"odds in double rounded once", "odds treated as none", "read value rounded to a short decimal"};
        for (int fault = 1; fault <= 3; fault++) {
            sr_derive_fault = fault;
            unlink("build/x_outd/metalsr.weights.params");
            rc = run_synth(&Y, "build/x_outd");
            sr_derive_fault = 0;
            printf("  control (%s): exit %d (must be 4)\n", names[fault - 1], rc);
            EXPECT(rc == 4 && access("build/x_outd/metalsr.weights.params", F_OK) != 0, "params-derive control '%s' did not fail (rc %d)", names[fault - 1], rc);
        }
        struct { const char *what; int rec, field, rc; } cases[] = {
            {"derivation code 5", 7, 37, 1}, {"code 3 with second index equal to the count", 3, 38, 1},
            {"code 4 on a stored fp64", 13, 36, 1}, {"code 1 on a zero", -1, 0, 3}, {"code 4 on x = 1", -2, 0, 3},
        };
        for (unsigned c = 0; c < sizeof cases / sizeof *cases; c++) {
            Spec s2 = sp; Synth Z;
            if (cases[c].rec == -1) { s2.stored[0] = 0.0; s2.expect[0] = 0.0; }               /* reciprocal of zero */
            if (cases[c].rec == -2) { s2.stored[13] = 1.0; s2.expect[13] = 0.0; }              /* odds of one */
            synth_build(&Z, &s2);
            if (cases[c].rec >= 0) {
                uint8_t *r = Z.loc + 16 + 24 * SR_NUM_TENSORS + 48 * cases[c].rec;
                if (cases[c].field == 38) { r[37] = 3; put16(r + 38, 16); } else if (cases[c].field == 37) r[37] = 5; else r[36] = 1;
            }
            rc = run_synth(&Z, NULL);
            printf("params-codes: %s: exit %d (expected %d)\n", cases[c].what, rc, cases[c].rc);
            EXPECT(rc == cases[c].rc, "params-codes: %s gave exit %d, expected %d", cases[c].what, rc, cases[c].rc);
            free(Z.pe); free(Z.want);
        }
        free(Y.pe); free(Y.want);
    }
    printf(fails ? "extract_test: %d FAILURES\n" : "extract_test: all ok\n", fails);
    return fails != 0;
}
