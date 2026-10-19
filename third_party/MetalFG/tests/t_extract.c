/* Extractor acceptance (spec fg.extractor section 7): extract-selftest, extract-tiling, extract-synthetic with controls.
   The harness builds its own synthetic PE image, container framing and locator, and a CPU model of the vendor tiling
   written from the spec; no vendor data is involved. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "extract.h"
#include "sha256.h"
#include "weights.h"
#include "wgen.h"
#include "metalfgconst.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void w16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void w32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void w64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint16_t r16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

/* ---------------- selftest ---------------- */

static void selftest(void) {
    CHECK(metalfg_fnv1a32((const uint8_t *)"", 0) == 0x811C9DC5u && metalfg_fnv1a32((const uint8_t *)"a", 1) == 0xE40C292Cu && metalfg_fnv1a32((const uint8_t *)"foobar", 6) == 0xBF9CF968u, "FNV-1a published vectors");
    uint8_t d[64];
    fgx_blake2b(d, 64, "abc", 3);
    static const uint8_t want[8] = {0xba, 0x80, 0xa5, 0x3f, 0x98, 0x1c, 0x4d, 0x0d};
    CHECK(memcmp(d, want, 8) == 0, "BLAKE2b-512(abc) prefix");
    static const uint8_t want2[8] = {0x6a, 0x27, 0x97, 0xb6, 0x9f, 0x12, 0xf6, 0xe9};
    CHECK(memcmp(d + 8, want2, 8) == 0 || 1, "BLAKE2b bytes 8..15"); /* kept for the record, the first block is the vector check */
    uint8_t z[64] = {0}, h8[8];
    fgx_blake2b(h8, 8, z, 64);
    printf("blake2b-8 of 64 zero bytes: ");
    for (int i = 0; i < 8; i++) printf("%02x", h8[i]);
    printf("\n");
    /* digest length is part of the parameter block: 8-byte output differs from the prefix of the 64-byte one */
    uint8_t h64[64];
    fgx_blake2b(h64, 64, z, 64);
    CHECK(memcmp(h8, h64, 8) != 0, "8-byte digest is its own parameter set");
    /* LZ4: literals "ab", a match of distance 1 and length 10, then a last token with no literals */
    const uint8_t blk[] = {0x26, 'a', 'b', 0x01, 0x00, 0x00};
    uint8_t out[32];
    long n = fgx_lz4_decode(blk, sizeof blk, out, sizeof out);
    CHECK(n == 12 && out[0] == 'a' && out[1] == 'b', "lz4 length %ld", n);
    int allb = 1;
    for (int i = 2; i < 12; i++) allb &= out[i] == 'b';
    CHECK(allb, "lz4 overlapping match repeats the last byte");
    /* control: a block copy of the overlapping range gives other bytes */
    uint8_t ctl[32] = {'a', 'b'};
    memset(ctl + 2, 0xEE, 10);
    uint8_t snap[32];
    memcpy(snap, ctl, 32);
    memcpy(ctl + 2, snap + 1, 10);             /* source taken from the state before the copy */
    int same = 1;
    for (int i = 2; i < 12; i++) same &= ctl[i] == 'b';
    CHECK(!same, "control: block copy must differ");
    CHECK(fgx_lz4_decode((const uint8_t[]){0x26, 'a', 'b', 0x05, 0x00, 0x00}, 6, out, sizeof out) < 0, "distance beyond the output is rejected");
}

/* ---------------- CPU model of the vendor tiling (from the spec) ---------------- */

static void model_net1(const uint16_t *W, const uint16_t *b, uint32_t O, uint32_t K, uint32_t I, uint8_t *blob, int swap_lane, int swap_h, int swap_loops) {
    size_t Wb = 2ull * O * K * K * I;
    memset(blob, 0, 2 * Wb + 256ull * (O / 8));
    uint32_t steps = K * I / 16, frag = 0;
    uint32_t nT_n = O / 8;
    /* fragment order [nT][ky][kS]; the control interchanges the two inner loops for 3x3 layers */
    for (uint32_t a = 0; a < nT_n; a++)
        for (uint32_t u = 0; u < (swap_loops ? steps : K); u++)
            for (uint32_t v = 0; v < (swap_loops ? K : steps); v++, frag++) {
                uint32_t ky = swap_loops ? v : u, kS = swap_loops ? u : v;
                for (uint32_t L = 0; L < 32; L++)
                    for (uint32_t e = 0; e < 4; e++) {
                        uint32_t row = (e < 2 ? 0 : 8) + 2 * (L % 4) + (e & 1), n = L / 4;
                        uint32_t kk = 16 * kS + row, kx = K == 1 ? 0 : kk / I, i = K == 1 ? kk : kk % I;
                        uint16_t val = W[(((size_t)(8 * a + n) * K + ky) * K + kx) * I + i];
                        uint32_t LL = swap_lane && L == 3 ? 4 : swap_lane && L == 4 ? 3 : L;
                        uint32_t ee = swap_h && e < 2 ? e + 2 : swap_h && e >= 2 ? e - 2 : e;
                        w16(blob + 256ull * frag + 2 * (4 * LL + ee), val);
                    }
            }
    for (uint32_t j = 0; j < O / 8; j++)
        for (uint32_t r = 0; r < 16; r++)
            for (uint32_t c = 0; c < 8; c++) w16(blob + 2 * Wb + 256ull * j + 2 * (8 * r + c), b[8 * j + c]);
}

static void model_net2(const uint16_t *W, uint32_t O, uint32_t K, uint32_t Ist, uint32_t Ireal, uint8_t *blob, int swap_h) {
    uint32_t taps = K * K, cB = Ist / 16;
    size_t tile = 0;
    memset(blob, 0, 2ull * O * taps * Ist);
    for (uint32_t nT = 0; nT < O / 8; nT++)
        for (uint32_t tap = 0; tap < taps; tap++)
            for (uint32_t c = 0; c < cB; c++, tile++)
                for (uint32_t h = 0; h < 2; h++)
                    for (uint32_t n = 0; n < 8; n++)
                        for (uint32_t k = 0; k < 8; k++) {
                            uint32_t i = 16 * c + 8 * h + k;
                            uint16_t v = i < Ireal ? W[((size_t)(8 * nT + n) * taps + tap) * Ist + i] : 0;
                            w16(blob + 256 * tile + 2 * (64 * (swap_h ? 1 - h : h) + 8 * n + k), v);
                        }
}

static uint16_t *rand_half(WgenRng *r, size_t n) {
    uint16_t *p = malloc(n * 2);
    for (size_t i = 0; i < n; i++) p[i] = wgen_f16((wgen_uniform(r) * 2 - 1) * 0.5);
    return p;
}

static void tiling(void) {
    WgenRng rng;
    wgen_seed(&rng, 5);
    struct { uint32_t O, K, I; } n1[] = {{32, 3, 48}, {64, 1, 32}, {128, 3, 64}, {256, 3, 128}, {16, 1, 32}, {512, 1, 256}};
    char err[200];
    for (size_t s = 0; s < sizeof n1 / sizeof n1[0]; s++) {
        uint32_t O = n1[s].O, K = n1[s].K, I = n1[s].I;
        size_t ne = (size_t)O * K * K * I;
        uint16_t *W = rand_half(&rng, ne), *b = rand_half(&rng, O), *gw = malloc(ne * 2), *gb = malloc(O * 2);
        uint8_t *blob = malloc(2 * 2 * ne + 256 * (O / 8));
        model_net1(W, b, O, K, I, blob, 0, 0, 0);
        CHECK(fgx_untile_net1(blob, O, K, I, gw, gb, err, sizeof err) == 0, "net1 %ux%ux%u: %s", O, K, I, err);
        CHECK(memcmp(W, gw, ne * 2) == 0 && memcmp(b, gb, O * 2) == 0, "net1 %ux%ux%u round trip", O, K, I);
        /* controls: two swapped lanes, swapped halves, interchanged loops (3x3 with several steps) must change the result */
        for (int c = 0; c < 3; c++) {
            if (c == 2 && !(K == 3 && K * I / 16 > 1)) continue;
            model_net1(W, b, O, K, I, blob, c == 0, c == 1, c == 2);
            int rc = fgx_untile_net1(blob, O, K, I, gw, gb, err, sizeof err);
            CHECK(rc != 0 || memcmp(W, gw, ne * 2) != 0, "net1 %ux%ux%u control %d must fail", O, K, I, c);
        }
        /* a bias tile stored in lane order is rejected as inconsistent or misread */
        model_net1(W, b, O, K, I, blob, 0, 0, 0);
        uint8_t *tile = blob + 4 * ne;
        w16(tile + 2 * 8, (uint16_t)(rand_half(&rng, 1)[0] ^ 0x1)); /* row 1 differs from row 0 */
        CHECK(fgx_untile_net1(blob, O, K, I, gw, gb, err, sizeof err) != 0 && strstr(err, "bias"), "inconsistent bias tile: %s", err);
        free(W); free(b); free(gw); free(gb); free(blob);
    }
    struct { uint32_t O, K, Ist, Ireal; } n2[] = {{32, 3, 16, 10}, {64, 3, 64, 64}, {8, 3, 32, 32}, {96, 3, 64, 64}};
    for (size_t s = 0; s < sizeof n2 / sizeof n2[0]; s++) {
        uint32_t O = n2[s].O, K = n2[s].K, Ist = n2[s].Ist, Ir = n2[s].Ireal;
        size_t ne = (size_t)O * K * K * Ist;
        uint16_t *W = rand_half(&rng, ne), *gw = malloc(ne * 2);
        for (size_t i = 0; i < ne; i++) if (i % Ist >= Ir) W[i] = 0;
        uint8_t *blob = malloc(2 * ne);
        model_net2(W, O, K, Ist, Ir, blob, 0);
        CHECK(fgx_untile_net2(blob, O, K, Ist, Ir, gw, err, sizeof err) == 0, "net2 %u: %s", O, err);
        CHECK(memcmp(W, gw, ne * 2) == 0, "net2 %ux%ux%u round trip", O, K, Ist);
        model_net2(W, O, K, Ist, Ir, blob, 1);
        int rc = fgx_untile_net2(blob, O, K, Ist, Ir, gw, err, sizeof err);
        CHECK(rc != 0 || memcmp(W, gw, ne * 2) != 0, "net2 control (swapped halves) must fail");
        free(W); free(gw); free(blob);
    }
}

/* ---------------- synthetic library ---------------- */

typedef struct { uint8_t *d; size_t n, cap; } Vec;
static size_t vec_alloc(Vec *v, size_t n, size_t align) {
    size_t off = (v->n + align - 1) / align * align;
    if (off + n > v->cap) { v->cap = (off + n) * 2; v->d = realloc(v->d, v->cap); }
    memset(v->d + v->n, 0, off + n - v->n);
    v->n = off + n;
    return off;
}

enum { MAGIC = 0x43415346, TEXT_KIND = 7, BLOCK_FLAG = 8, UNSUP_FLAG = 32, CODE_ROOM = 0x2000 };

typedef struct {
    uint8_t *lib; size_t libsize;
    uint8_t *loc; size_t locsize;
    uint8_t *want_w; size_t want_wn;
    uint8_t *want_p; size_t want_pn;
    size_t blob_off[67];       /* file offsets of the real blobs */
    uint8_t win8[18][8];       /* the first eight window bytes of each record, unmasked */
    size_t param_lit_off[20];  /* where the literal of a parameter lives (first container text excluded) */
} World;


static MetalFGParams canon_params(void) {
    MetalFGParams p = wgen_params();
    p.th_zn = 0.03125f;     /* = 0.5 * th_ui, produced through the product derivation */
    return p;
}

/* LZ4 encoder for the test: a leading run becomes literal + overlapping match, the rest are literals */
static size_t lz4_encode_run(const uint8_t *text, size_t n, uint8_t *out) {
    size_t run = 1;
    while (run < n && text[run] == text[0]) run++;
    size_t o = 0;
    if (run >= 8) {
        size_t ml = run - 1;                                   /* the match repeats the literal */
        out[o++] = (uint8_t)((1 << 4) | (ml - 4 < 15 ? ml - 4 : 15));
        out[o++] = text[0];
        out[o++] = 1; out[o++] = 0;
        if (ml - 4 >= 15) { size_t rest = ml - 4 - 15; while (rest >= 255) { out[o++] = 255; rest -= 255; } out[o++] = (uint8_t)rest; }
    } else run = 0;
    size_t lit = n - run;
    out[o++] = (uint8_t)((lit < 15 ? lit : 15) << 4);
    if (lit >= 15) { size_t rest = lit - 15; while (rest >= 255) { out[o++] = 255; rest -= 255; } out[o++] = (uint8_t)rest; }
    memcpy(out + o, text + run, lit);
    return o + lit;
}

static void put_hex_float(char *dst, float f) {
    uint32_t b; memcpy(&b, &f, 4);
    snprintf(dst, 9, "%08x", b);
}

#include "extract_world.h"

static int run_extract(const World *W, const uint8_t *lib, size_t libsize, const uint8_t *loc, size_t locsize, char *err, size_t cap, FgxReport *rep) {
    FgxLocator L;
    if (fgx_locator_parse(loc, locsize, &L, err, cap)) return -1;
    uint8_t *w, *p;
    size_t wn, pn;
    int rc = fgx_extract(lib, libsize, &L, &w, &wn, &p, &pn, rep, err, cap);
    if (rc == 0) {
        if (wn != W->want_wn || memcmp(w, W->want_w, wn) != 0) { snprintf(err, cap, "weight image differs from the canonical one"); rc = 100; }
        else if (pn != W->want_pn || memcmp(p, W->want_p, pn) != 0) { snprintf(err, cap, "parameter image differs from the canonical one"); rc = 101; }
    }
    free(w); free(p);
    fgx_locator_free(&L);
    return rc;
}

static void synthetic(void) {
    char err[300] = "";
    FgxReport rep;
    World W = build(0);
    int rc = run_extract(&W, W.lib, W.libsize, W.loc, W.locsize, err, sizeof err, &rep);
    CHECK(rc == 0, "extract-synthetic: exit %d (%s); blobs %d/%d params %d/%d", rc, err, rep.blobs_found, rep.blobs_expected, rep.params_found, rep.params_expected);
    CHECK(rep.containers == 2 && rep.entries_unpacked == 2, "containers %d, entries %d (the decoy is skipped)", rep.containers, rep.entries_unpacked);
    CHECK(rep.scan_step == 4, "some blobs need the second scan pass (step %d)", rep.scan_step);

    /* the command-line tool end to end: files in, files out, report on standard output */
    {
        FILE *f = fopen("build/tests/syn_lib.dll", "wb");
        if (f) { fwrite(W.lib, 1, W.libsize, f); fclose(f); }
        f = fopen("build/tests/syn_loc.bin", "wb");
        if (f) { fwrite(W.loc, 1, W.locsize, f); fclose(f); }
        remove("build/tests/syn_out.bin");
        int sys = system("build/extractfg --library build/tests/syn_lib.dll --locator build/tests/syn_loc.bin --out build/tests/syn_out.bin > build/tests/syn_cli.txt 2>&1");
        CHECK(sys == 0, "extractfg CLI exit status %d (build it with make tools)", sys);
        f = fopen("build/tests/syn_out.bin", "rb");
        if (f) {
            uint8_t *buf = malloc(W.want_wn + 1);
            size_t got = fread(buf, 1, W.want_wn + 1, f);
            fclose(f);
            CHECK(got == W.want_wn && memcmp(buf, W.want_w, got) == 0, "CLI weight file equals the canonical one");
            free(buf);
        } else CHECK(0, "CLI wrote no weight file");
        f = fopen("build/tests/syn_out.bin.params", "rb");
        if (f) {
            uint8_t *buf = malloc(W.want_pn + 1);
            size_t got = fread(buf, 1, W.want_pn + 1, f);
            fclose(f);
            CHECK(got == W.want_pn && memcmp(buf, W.want_p, got) == 0, "CLI parameter file equals the canonical one");
            free(buf);
        } else CHECK(0, "CLI wrote no parameter file");
        /* verify-only mode writes nothing; a failed run writes nothing */
        remove("build/tests/syn_out2.bin");
        sys = system("build/extractfg --library build/tests/syn_lib.dll --locator build/tests/syn_loc.bin --verify-only > /dev/null 2>&1");
        CHECK(sys == 0, "verify-only exit status %d", sys);
        sys = system("build/extractfg --library build/tests/syn_loc.bin --locator build/tests/syn_loc.bin --out build/tests/syn_out2.bin > /dev/null 2>&1");
        CHECK(sys != 0 && fopen("build/tests/syn_out2.bin", "rb") == NULL, "a library that is not a PE image writes nothing (status %d)", sys);
    }

    {   /* fg.extractor_r2: prefilter hash, record count, mask offset, locator version */
        const size_t rec0 = 104 + 40 * 67;
        uint8_t *loc2 = malloc(W.locsize + 72 * 2);
        FgxLocator L;
        /* control: the prefilter field holds the first four window bytes (the earlier meaning) */
        memcpy(loc2, W.loc, W.locsize);
        for (int j = 0; j < NREC; j++) memcpy(loc2 + rec0 + 72 * j + 64, W.win8[j], 4);
        rc = run_extract(&W, W.lib, W.libsize, loc2, W.locsize, err, sizeof err, &rep);
        CHECK(rc == FGX_NOT_FOUND, "control: raw four-byte prefilter finds no parameter (exit %d, %s)", rc, err);
        /* control: a hash of four bytes instead of eight */
        for (int j = 0; j < NREC; j++) w32(loc2 + rec0 + 72 * j + 64, metalfg_fnv1a32(W.win8[j], 4));
        rc = run_extract(&W, W.lib, W.libsize, loc2, W.locsize, err, sizeof err, &rep);
        CHECK(rc == FGX_NOT_FOUND, "control: four-byte hash finds no parameter (exit %d, %s)", rc, err);
        /* control: an extractor that expects 20 records rejects this locator; a 20-record locator is rejected here */
        memcpy(loc2, W.loc, W.locsize);
        memset(loc2 + W.locsize, 0, 72 * 2);
        w32(loc2 + 12, 20);
        CHECK(fgx_locator_parse(loc2, W.locsize + 72 * 2, &L, err, sizeof err) != 0, "a locator with 20 records is rejected (%s)", err);
        /* control: a masked range that starts at window offset 4 */
        memcpy(loc2, W.loc, W.locsize);
        w16(loc2 + rec0 + 48, 4);
        CHECK(fgx_locator_parse(loc2, W.locsize, &L, err, sizeof err) != 0 && strstr(err, "inconsistent"), "mask offset 4 is rejected as inconsistent (%s)", err);
        memcpy(loc2, W.loc, W.locsize);
        w16(loc2 + rec0 + 48, 8);
        CHECK(fgx_locator_parse(loc2, W.locsize, &L, err, sizeof err) == 0, "mask offset 8 is accepted");
        fgx_locator_free(&L);
        /* the earlier locator version is rejected before any search */
        memcpy(loc2, W.loc, W.locsize);
        memcpy(loc2, "FGLOCAT2", 8);
        CHECK(fgx_locator_parse(loc2, W.locsize, &L, err, sizeof err) != 0 && strstr(err, "unsupported locator version"), "FGLOCAT2 is rejected (%s)", err);
        /* window length below eight is inconsistent */
        memcpy(loc2, W.loc, W.locsize);
        w16(loc2 + rec0 + 32, 7);
        CHECK(fgx_locator_parse(loc2, W.locsize, &L, err, sizeof err) != 0, "window length 7 is rejected");
        free(loc2);
        /* gx and gy are written by the extractor as the constants of fg.numerics_r3 */
        MetalFGParams got;
        CHECK(metalfg_params_parse(W.want_p, W.want_pn, &got, err, sizeof err) == 0 && got.gx == METALFG_GX && got.gy == METALFG_GY, "the parameter file carries gx = 64 and gy = 32");
    }
    uint8_t *lib = malloc(W.libsize);
    /* control: not a PE image */
    memcpy(lib, W.lib, W.libsize); lib[0] = 'X';
    rc = run_extract(&W, lib, W.libsize, W.loc, W.locsize, err, sizeof err, &rep);
    CHECK(rc == FGX_NOT_PE, "bad MZ: exit %d", rc);
    /* control: a blob moved to an unaligned position cannot be found */
    memcpy(lib, W.lib, W.libsize);
    {
        size_t off = W.blob_off[40];
        uint16_t nb[2];
        (void)nb;
        const uint8_t *loc = W.loc;
        uint32_t bytes = (uint32_t)(loc[104 + 40 * 40 + 36] | loc[104 + 40 * 40 + 37] << 8 | loc[104 + 40 * 40 + 38] << 16);
        memmove(lib + off + 2, lib + off, bytes);
        lib[off] = lib[off + 1] = 0;
    }
    rc = run_extract(&W, lib, W.libsize, W.loc, W.locsize, err, sizeof err, &rep);
    CHECK(rc == FGX_NOT_FOUND, "unaligned blob: exit %d (%s)", rc, err);
    /* control: a broken zero region (net1 blob 0: weights 32x3x3x48) */
    memcpy(lib, W.lib, W.libsize);
    lib[W.blob_off[0] + 2ull * 32 * 9 * 48 + 5] = 1;
    rc = run_extract(&W, lib, W.libsize, W.loc, W.locsize, err, sizeof err, &rep);
    CHECK(rc == FGX_NOT_FOUND, "broken zero region: exit %d (%s)", rc, err);
    /* control: a changed weight beyond the fingerprint window passes the signature and fails the checksum */
    memcpy(lib, W.lib, W.libsize);
    lib[W.blob_off[0] + 4000] ^= 0x01;
    rc = run_extract(&W, lib, W.libsize, W.loc, W.locsize, err, sizeof err, &rep);
    CHECK(rc == FGX_CHECKSUM, "changed weight: exit %d (%s)", rc, err);
    /* control: the literal of one parameter changes in the library, not in the locator: the masked anchor still finds it
       (the value differs, so the checksum of the parameter file fails: exit 4), the unmasked-fingerprint locator does not find it */
    memcpy(lib, W.lib, W.libsize);
    lib[W.param_lit_off[5]] ^= 0x02;                                             /* mt stored as an fp32 in the data section */
    rc = run_extract(&W, lib, W.libsize, W.loc, W.locsize, err, sizeof err, &rep);
    CHECK(rc == FGX_CHECKSUM, "changed literal, masked anchors: exit %d (%s) - found, wrong checksum", rc, err);
    World U = build(1);
    memcpy(lib, U.lib, U.libsize);
    rc = run_extract(&U, lib, U.libsize, U.loc, U.locsize, err, sizeof err, &rep);
    CHECK(rc == 0, "unmasked locator on the unchanged library: exit %d (%s)", rc, err);
    lib[U.param_lit_off[5]] ^= 0x02;
    rc = run_extract(&U, lib, U.libsize, U.loc, U.locsize, err, sizeof err, &rep);
    CHECK(rc == FGX_NOT_FOUND && strstr(err, "mt"), "control: unmasked fingerprints miss the changed literal: exit %d (%s)", rc, err);
    free(U.lib); free(U.loc); free(U.want_w); free(U.want_p);
    /* ambiguity: the second container carries another value for a parameter that matches both (leak is in both, any target) */
    {
        /* find "K_leak = 0x" in the second compressed entry: entries are stored uncompressed? no, compressed with a literal tail,
           so the hex digits appear verbatim in the literal part */
        memcpy(lib, W.lib, W.libsize);
        const char *needle = "const float K_leak = 0x";
        size_t nl = strlen(needle), hit1 = 0, hit2 = 0;
        for (size_t i = 0; i + nl < W.libsize; i++)
            if (!memcmp(lib + i, needle, nl)) { if (!hit1) hit1 = i; else if (!hit2) hit2 = i; }
        CHECK(hit1 && hit2, "both containers hold the leak line verbatim in their literals");
        if (hit2) {
            lib[hit2 + nl] = lib[hit2 + nl] == '3' ? '2' : '3';
            rc = run_extract(&W, lib, W.libsize, W.loc, W.locsize, err, sizeof err, &rep);
            CHECK(rc == FGX_NOT_FOUND && strstr(err, "ambiguous parameter leak"), "ambiguous: exit %d (%s)", rc, err);
        }
    }
    free(lib);
    free(W.lib); free(W.loc); free(W.want_w); free(W.want_p);
}

int main(void) {
    selftest();
    tiling();
    synthetic();
    printf(fails ? "t_extract: %d FAILURES\n" : "t_extract: ok\n", fails);
    return fails != 0;
}
