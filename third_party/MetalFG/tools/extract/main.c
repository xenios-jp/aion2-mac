/* extractfg: setup-time extractor (spec fg.extractor).
   usage: extractfg --library LIB --locator LOC --out WEIGHTS [--verify-only] [--discover] [-v]
   Writes WEIGHTS and WEIGHTS.params. Exit: 0 ok, 1 usage or I/O, 2 not a PE image, 3 not found / ambiguous / inconsistent,
   4 checksum mismatch. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "extract.h"
#include "sha256.h"
#include "weights.h"

static uint8_t *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    uint8_t *b = malloc((size_t)sz + 1);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static void hex(const uint8_t *d, char *out) { for (int i = 0; i < 32; i++) sprintf(out + 2 * i, "%02x", d[i]); }

int main(int argc, char **argv) {
    const char *lib = NULL, *loc = NULL, *out = NULL;
    int verify = 0, discover = 0, verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--library") && i + 1 < argc) lib = argv[++i];
        else if (!strcmp(argv[i], "--locator") && i + 1 < argc) loc = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--verify-only")) verify = 1;
        else if (!strcmp(argv[i], "--discover")) discover = 1;
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown or incomplete option '%s'\n", argv[i]); return FGX_USAGE; }
    }
    if (!lib || !loc || (!out && !verify && !discover)) {
        fprintf(stderr, "usage: extractfg --library LIB --locator LOC (--out WEIGHTS | --verify-only | --discover) [-v]\n");
        return FGX_USAGE;
    }
    size_t ln, cn;
    uint8_t *L = slurp(lib, &ln), *C = slurp(loc, &cn);
    if (!L) { fprintf(stderr, "cannot read library '%s'\n", lib); return FGX_USAGE; }
    if (!C) { fprintf(stderr, "cannot read locator '%s'\n", loc); return FGX_USAGE; }
    char err[400] = "";
    FgxLocator lc;
    if (fgx_locator_parse(C, cn, &lc, err, sizeof err)) { fprintf(stderr, "%s\n", err); return FGX_USAGE; }
    if (discover) {
        char *t = fgx_discover(L, ln, &lc);
        fputs(t, stdout);
        free(t);
        return FGX_OK;
    }
    uint8_t *w = NULL, *p = NULL;
    size_t wn = 0, pn = 0;
    FgxReport rep;
    int rc = fgx_extract(L, ln, &lc, &w, &wn, &p, &pn, &rep, err, sizeof err);
    char h[65];
    hex(rep.lib_sha, h);
    printf("library=%s\nsize=%zu\nsha256=%s\n", lib, ln, h);
    printf("blobs=%d/%d\nparameters=%d/%d\ncontainers=%d\nentries_unpacked=%d\nentries_skipped=%d\nscan_step=%d\n", rep.blobs_found, rep.blobs_expected,
           rep.params_found, rep.params_expected, rep.containers, rep.entries_unpacked, rep.entries_skipped, rep.scan_step);
    if (rc == FGX_OK) { hex(rep.payload_sha, h); printf("payload_sha256=%s\n", h); }
    if (rc) { printf("status=failed\n"); fprintf(stderr, "%s\n", err); return rc; }
    if (!verify) {
        char pp[1100];
        snprintf(pp, sizeof pp, "%s.params", out);
        if (metalfg_write_file_atomic(out, w, wn, err, sizeof err) || metalfg_write_file_atomic(pp, p, pn, err, sizeof err)) {
            fprintf(stderr, "%s\n", err);
            printf("status=failed\n");
            return FGX_USAGE;
        }
    }
    if (verbose) fprintf(stderr, "weights %zu bytes, parameters %zu bytes\n", wn, pn);
    printf("status=ok\n");
    return FGX_OK;
}
