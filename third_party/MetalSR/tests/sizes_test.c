/* Size contract and size query (sr.test_scenes_r6): metalsr_sizes against formulas written out independently, the exhaustive sweeps, and a control. */
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "metalsr.h"

static int bad;
#define CHECK(c, ...) do { if (!(c)) { if (bad < 20) { printf("  FAIL %s: ", #c); printf(__VA_ARGS__); printf("\n"); } bad++; } } while (0)

static MetalSRCreateInfo mk(unsigned wo, unsigned ho, unsigned mw, unsigned mh) {
    MetalSRCreateInfo i;
    memset(&i, 0, sizeof i);
    i.version = METALSR_INTERFACE_VERSION; i.out_w = wo; i.out_h = ho; i.max_rw = mw; i.max_rh = mh;
    return i;
}
static unsigned ceil_to(unsigned v, unsigned m) { return (v + m - 1) / m * m; }
static unsigned grid(unsigned wo, unsigned wr, unsigned nmin) { unsigned l = wo / 2 <= wr ? wo / 2 : wr / 2; return l > nmin ? l : nmin; }

/* expected record, from the table of the spec; pad is 128 (the control uses 64) */
static void expect(const MetalSRCreateInfo *i, unsigned rw, unsigned rh, double rwide, unsigned nmin, unsigned pad, MetalSRSizes *e) {
    memset(e, 0, sizeof *e);
    e->wp = ceil_to(i->out_w, pad); e->hp = ceil_to(i->out_h, pad);
    for (int k = 0; k < 6; k++) { e->lw[k] = e->wp >> (k + 2); e->lh[k] = e->hp >> (k + 2); }
    e->path = (double)i->out_w / (double)i->max_rw > rwide;
    e->sx = (float)((double)rw / (double)i->out_w); e->sy = (float)((double)rh / (double)i->out_h);
    e->gw = grid(i->out_w, rw, nmin); e->gh = grid(i->out_h, rh, nmin);
}

static int run(const MetalSRCreateInfo *i, unsigned rw, unsigned rh, double rwide, unsigned nmin, MetalSRSizes *o, char *err) {
    memset(o, 0xAB, sizeof *o); err[0] = 0;
    return metalsr_sizes(i, rw, rh, rwide, nmin, o, err, 256);
}

int main(void) {
    MetalSRSizes o, e;
    char err[256];
    const double rwide = 2.4;
    const unsigned nmin = 16;
    CHECK(sizeof(MetalSRSizes) == 80, "size");
    CHECK(offsetof(MetalSRSizes, lw) == 8 && offsetof(MetalSRSizes, lh) == 32 && offsetof(MetalSRSizes, path) == 56 && offsetof(MetalSRSizes, sx) == 60 &&
          offsetof(MetalSRSizes, sy) == 64 && offsetof(MetalSRSizes, gw) == 68 && offsetof(MetalSRSizes, gh) == 72 && offsetof(MetalSRSizes, reserved_) == 76, "offsets");

    /* the probes of the spec: create must fail naming the field (metalsr_sizes writes the message create writes) */
    struct { unsigned wo, ho, mw, mh; const char *field; } probe[] = {
        {16385, 1024, 16385, 1024, "out_w"}, {1024, 16385, 1024, 16385, "out_h"}, {1920, 1080, 1921, 1080, "max_rw"}, {1920, 1080, 1920, 1081, "max_rh"},
        {0, 1080, 1, 1, "out_w"}, {1920, 0, 1, 1, "out_h"}, {1920, 1080, 0, 1, "max_rw"}, {1920, 1080, 1, 0, "max_rh"}};
    for (size_t p = 0; p < sizeof probe / sizeof *probe; p++) {
        MetalSRCreateInfo i = mk(probe[p].wo, probe[p].ho, probe[p].mw, probe[p].mh);
        int rc = run(&i, 1, 1, rwide, nmin, &o, err);
        CHECK(rc == 1 && strstr(err, probe[p].field), "probe %zu: rc %d '%s'", p, rc, err);
        CHECK(((unsigned char *)&o)[0] == 0xAB, "record changed on failure");
        printf("  probe %-7s out %ux%u max render %ux%u -> rc %d: %s\n", probe[p].field, probe[p].wo, probe[p].ho, probe[p].mw, probe[p].mh, rc, err);
    }
    { MetalSRCreateInfo i = mk(1920, 1080, 1280, 720); i.version = 1; CHECK(run(&i, 720, 480, rwide, nmin, &o, err) == 1 && err[0], "version"); }
    { MetalSRCreateInfo i = mk(16384, 16384, 16384, 16384); CHECK(run(&i, 16384, 16384, rwide, nmin, &o, err) == 0, "caps accepted: %s", err); }
    { MetalSRCreateInfo i = mk(1920, 1080, 1280, 720);
      CHECK(run(&i, 1281, 720, rwide, nmin, &o, err) == 2 && strstr(err, "rw"), "x-call-over '%s'", err);
      CHECK(run(&i, 1280, 721, rwide, nmin, &o, err) == 2 && strstr(err, "rh"), "rh over '%s'", err); }

    /* whole record, both paths */
    unsigned S[][6] = {{1920, 1080, 1280, 720, 1280, 720}, {1920, 1080, 640, 360, 640, 360}, {65, 65, 65, 65, 33, 17}, {1, 1, 1, 1, 1, 1},
                       {3840, 2160, 1920, 1080, 1000, 700}, {2175, 2751, 1000, 1000, 28, 821}, {128, 128, 128, 128, 128, 128}, {129, 127, 60, 50, 60, 50}};
    for (size_t s = 0; s < sizeof S / sizeof *S; s++)
        for (int pass = 0; pass < 2; pass++) {
            MetalSRCreateInfo i = mk(S[s][0], S[s][1], S[s][2], S[s][3]);
            double rw_ = pass ? 1.0 : 100.0;
            int rc = run(&i, S[s][4], S[s][5], rw_, nmin, &o, err);
            expect(&i, S[s][4], S[s][5], rw_, nmin, 128, &e);
            CHECK(rc == 0 && !memcmp(&o, &e, sizeof o), "record %zu pass %d rc %d", s, pass, rc);
        }

    /* exhaustive sweeps (size-exhaustive): widths; the height axis is the same code with the roles swapped, checked on the verdicts */
    for (unsigned wo = 1; wo <= 16384; wo++) {
        for (unsigned mw = 0; mw <= wo + 1; mw++) {
            MetalSRCreateInfo j = mk(wo, 1, mw, 1);
            int rc = run(&j, mw, 1, rwide, nmin, &o, err);
            CHECK(rc == ((mw == 0 || mw == wo + 1) ? 1 : 0), "wo %u max_rw %u rc %d", wo, mw, rc);
            if (!rc) { expect(&j, mw, 1, rwide, nmin, 128, &e); CHECK(!memcmp(&o, &e, sizeof o), "record wo %u mw %u", wo, mw); }
            MetalSRCreateInfo h = mk(1, wo, 1, mw);
            rc = run(&h, 1, mw, rwide, nmin, &o, err);
            CHECK(rc == ((mw == 0 || mw == wo + 1) ? 1 : 0), "ho %u max_rh %u rc %d", wo, mw, rc);
            if (!rc) { expect(&h, 1, mw, rwide, nmin, 128, &e); CHECK(!memcmp(&o, &e, sizeof o), "record ho %u mh %u", wo, mw); }
        }
        MetalSRCreateInfo k = mk(wo, wo, wo, wo);
        for (unsigned r = 0; r <= wo + 1; r += (r < 3 || wo + 1 - r < 3 ? 1 : 1 + wo / 5)) {
            int rc = run(&k, r, 1, rwide, nmin, &o, err);
            CHECK(rc == ((r == 0 || r == wo + 1) ? 2 : 0), "wo %u rw %u rc %d", wo, r, rc);
            if (!rc) { expect(&k, r, 1, rwide, nmin, 128, &e); CHECK(!memcmp(&o, &e, sizeof o), "record wo %u rw %u", wo, r); }
            rc = run(&k, 1, r, rwide, nmin, &o, err);
            CHECK(rc == ((r == 0 || r == wo + 1) ? 2 : 0), "ho %u rh %u rc %d", wo, r, rc);
        }
    }
    { MetalSRCreateInfo i = mk(16385, 1, 1, 1); CHECK(run(&i, 1, 1, rwide, nmin, &o, err) == 1, "above cap"); }

    /* control: a query that pads to 64 instead of 128 gives a different Wp where the two roundings differ */
    { MetalSRCreateInfo i = mk(129, 129, 129, 129);
      run(&i, 129, 129, rwide, nmin, &o, err); expect(&i, 129, 129, rwide, nmin, 64, &e);
      printf("  control (pad 64): wp %u vs %u -> %s\n", o.wp, e.wp, o.wp != e.wp ? "detected" : "NOT detected");
      CHECK(o.wp != e.wp, "padding to 64 must be detected"); }
    printf(bad ? "sizes: FAIL (%d)\n" : "sizes: ok\n", bad);
    return bad != 0;
}
