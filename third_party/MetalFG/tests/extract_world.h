/* Builds the synthetic library, locator and canonical outputs for the extractor tests (included by t_extract.c). */

enum { PRM_PREFIX_MAX = 64, NREC = 18 };
/* record j describes parameter-file entry PIDX[j]: every entry except gx (3) and gy (4) (fg.extractor_r2) */
static const int PIDX[NREC] = {0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

/* How each of the 20 parameters is stored in the synthetic library (areas, stored forms and derivations of fg.extractor) */
typedef struct { int area, type, deriv, second, target; double stored; } PForm;

static void param_forms(const MetalFGParams *P, PForm f[METALFG_PARAM_COUNT]) {
    double v[METALFG_PARAM_COUNT] = {P->hm, P->hf, P->hb, P->gx, P->gy, P->mt, P->mfa, P->mfh, P->th_ui, P->th_uia, P->leak, P->th_mask,
                                 P->b_steer, P->sat_n2, P->d_dyn, P->d_sim, P->th_cam, P->sp_min, P->sp_max, P->th_zn};
    for (int i = 0; i < METALFG_PARAM_COUNT; i++) f[i] = (PForm){-1, 0, 0, 65535, 0, v[i]};
    f[0] = (PForm){1, 2, 0, 65535, 0, v[0]};                         /* int32 immediate in code */
    f[1] = (PForm){1, 2, 4, 65535, 0, v[1] - 1};                     /* count minus one, addend +1 */
    f[2] = (PForm){1, 2, 0, 65535, 0, v[2]};                         /* int32 behind a relative reference */
    f[3] = (PForm){0, 2, 0, 65535, 0, v[3]};                         /* int32 in a data section */
    f[4] = (PForm){0, 2, 0, 65535, 0, v[4]};
    f[5] = (PForm){0, 0, 0, 65535, 0, v[5]};                         /* fp32 in a data section */
    f[6] = (PForm){1, 0, 0, 65535, 0, v[6]};                         /* fp32 behind a relative reference */
    f[7] = (PForm){2, 5, 0, 65535, 1, v[7]};                         /* hexadecimal text, first container only */
    f[8] = (PForm){2, 5, 0, 65535, 0, v[8]};
    f[9] = (PForm){0, 0, 0, 65535, 0, v[9]};
    f[10] = (PForm){2, 5, 0, 65535, 0, v[10]};                       /* present in both containers */
    f[11] = (PForm){2, 5, 0, 65535, 2, v[11]};
    f[12] = (PForm){2, 5, 1, 65535, 0, 1.0 / v[12]};                 /* reciprocal */
    f[13] = (PForm){2, 5, 2, 65535, 0, -v[13]};                      /* negation */
    f[14] = (PForm){2, 6, 0, 65535, 0, v[14]};                       /* decimal text */
    f[15] = (PForm){2, 6, 0, 65535, 1, v[15]};
    f[16] = (PForm){1, 0, 0, 65535, 0, v[16]};                       /* fp32 immediate in code */
    f[17] = (PForm){2, 6, 0, 65535, 0, v[17]};
    f[18] = (PForm){2, 6, 0, 65535, 0, v[18]};
    f[19] = (PForm){2, 5, 3, 6, 0, 0.5};                             /* product with the value of record 6 (th_ui) */
}

static World build(int unmasked_fp) {
    World W;
    memset(&W, 0, sizeof W);
    WgenRng rng;
    wgen_seed(&rng, 2024);
    (void)rng;
    const MetalFGTensorInfo *T = metalfg_weight_table();
    uint16_t *tens[METALFG_TENSOR_COUNT];
    wgen_tensors(WGEN_RANDOM, 11, tens);
    size_t n;
    W.want_w = metalfg_weights_image((const uint16_t *const *)tens, &n);
    W.want_wn = n;
    MetalFGParams P = canon_params();
    W.want_p = metalfg_params_image(&P, &W.want_pn);

    Vec data = {0};
    uint8_t blobrec[67 * 40] = {0}, parrec[NREC * 72] = {0};
    size_t blob_rel[67];
    int nb = 0;

    /* ---- blobs: most at 16-byte alignment, some only at 4 (second scan pass), decoys in front of some ---- */
    for (int t = 0; t < METALFG_TENSOR_COUNT;) {
        const MetalFGTensorInfo *w = &T[t];
        int net1 = strncmp(w->name, "n1.", 3) == 0;
        uint32_t O = w->O, K = w->kh, Ist = w->I, Ireal = Ist;
        int kind = net1 ? 0 : 1, variant = 0;
        if (!strcmp(w->name, "n2.a.d0.w")) Ireal = 10;
        if (!strcmp(w->name, "n2.b.d0.w")) variant = 2;
        size_t ne = (size_t)O * K * K * Ist;
        size_t bytes = net1 ? 4 * ne + 256 * (O / 8) : 2 * ne;
        uint8_t *blob = calloc(1, bytes);
        if (net1) model_net1(tens[t], tens[t + 1], O, K, Ist, blob, 0, 0, 0);
        else if (variant == 2) memcpy(blob, tens[t], bytes);
        else model_net2(tens[t], O, K, Ist, Ireal, blob, 0);
        if (nb % 7 == 2 && (kind == 0 || variant == 0)) {
            /* decoy: the right first 256 bytes, then non-finite filler, so the fingerprint hits and the signature fails */
            size_t dec = vec_alloc(&data, 256, 16);
            memcpy(data.d + dec, blob, 256);
            size_t fill = vec_alloc(&data, 16, 16);
            memset(data.d + fill, 0x7c, 16);
        }
        size_t off;
        if (nb % 5 == 3) { vec_alloc(&data, 4, 16); off = vec_alloc(&data, bytes, 4); }      /* offset = 4 mod 16 */
        else off = vec_alloc(&data, bytes, 16);
        memcpy(data.d + off, blob, bytes);
        blob_rel[nb] = off;
        uint8_t *r = blobrec + 40 * nb;
        w32(r + 8, (uint32_t)t); w32(r + 12, (uint32_t)kind); w32(r + 16, O); w32(r + 20, K); w32(r + 24, Ist); w32(r + 28, Ireal);
        w32(r + 32, (uint32_t)variant); w32(r + 36, (uint32_t)bytes);
        free(blob);
        nb++;
        if (net1) { t += 2; continue; }
        t++;                                                    /* the bias of this net2 layer is its own record */
        const MetalFGTensorInfo *bt = &T[t];
        size_t bb = 2ull * bt->O, boff = vec_alloc(&data, bb, 4);
        memcpy(data.d + boff, tens[t], bb);
        blob_rel[nb] = boff;
        vec_alloc(&data, 64, 16);                               /* the 64-byte window of a short blob reads into this padding */
        r = blobrec + 40 * nb;
        w32(r + 8, (uint32_t)t); w32(r + 12, 2); w32(r + 16, bt->O); w32(r + 20, 1); w32(r + 24, 1); w32(r + 28, 1); w32(r + 36, (uint32_t)bb);
        nb++;
        t++;
    }

    /* ---- parameters ---- */
    static const char *names[METALFG_PARAM_COUNT] = {"hm", "hf", "hb", "gx", "gy", "mt", "mfa", "mfh", "th_ui", "th_uia", "leak", "th_mask",
                                                 "b_steer", "sat_n2", "d_dyn", "d_sim", "th_cam", "sp_min", "sp_max", "th_zn"};
    PForm pf[METALFG_PARAM_COUNT];
    param_forms(&P, pf);
    uint8_t code[CODE_ROOM] = {0};
    size_t code_pos = 64;
    char text[8192];
    size_t tl = 0;
    memset(text, 'a', 24);                                      /* a leading run: encoded as a literal and an overlapping match */
    tl = 24;
    text[tl++] = '\n';
    size_t code_at[METALFG_PARAM_COUNT] = {0}, data_at[METALFG_PARAM_COUNT] = {0}, pool_at[METALFG_PARAM_COUNT] = {0};
    size_t win_text_at[METALFG_PARAM_COUNT] = {0};
    size_t lit_in_code[METALFG_PARAM_COUNT] = {0};
    for (int j = 0; j < NREC; j++) {
        int i = PIDX[j];
        uint8_t *r = parrec + 72 * j;
        memcpy(r, names[i], strlen(names[i]));
        r[36] = (uint8_t)pf[i].type; r[37] = (uint8_t)pf[i].deriv; w16(r + 38, (uint16_t)pf[i].second);
        r[40] = (uint8_t)pf[i].area;
        w32(r + 44, (uint32_t)pf[i].target);
        if (pf[i].deriv == 4) w32(r + 60, 1);
        float fv = (float)pf[i].stored;
        uint32_t fb, valbits;
        memcpy(&fb, &fv, 4);
        valbits = pf[i].type == 0 ? fb : (uint32_t)(int32_t)pf[i].stored;
        uint16_t wlen = 0, mo = 0, ml = 0;
        int16_t disp = 0;
        if (pf[i].area == 0) {
            size_t off = vec_alloc(&data, 24, 4);
            uint8_t *p = data.d + off;
            p[0] = 'F'; p[1] = 'G'; p[2] = 'P'; p[3] = (uint8_t)('A' + i);
            memcpy(p + 4, "-key", 4);
            w32(p + 8, valbits);
            memcpy(p + 12, "-end-end", 8);
            data_at[i] = off;
            wlen = 20; mo = 8; ml = 4; disp = 8;
        } else if (pf[i].area == 1) {
            uint8_t *p = code + code_pos;
            code_at[i] = code_pos;
            code_pos += 16;
            if (i == 2 || i == 6) {                             /* [4 fixed][8B 05][rel32][C3]: a relative reference into the data section */
                size_t pool = vec_alloc(&data, 4, 4);
                w32(data.d + pool, valbits);
                pool_at[i] = pool;
                p[0] = 0x0F; p[1] = 0x1F; p[2] = 0x45; p[3] = (uint8_t)i; p[4] = 0x8B; p[5] = 0x05; p[6] = p[7] = 0x90; p[12] = 0xC3;
                wlen = 13; mo = 8; ml = 4; disp = 8;
                r[41] = 1; r[42] = 1;                           /* indirect, one instruction byte after the reference */
            } else {                                            /* [4 fixed][48 C7 C0][imm32][90 90] */
                p[0] = 0x0F; p[1] = 0x1F; p[2] = 0x44; p[3] = (uint8_t)i; p[4] = 0x48; p[5] = 0xC7; p[6] = 0xC0; p[7] = 0x90;
                w32(p + 8, valbits);
                p[12] = 0x90; p[13] = 0x90;
                wlen = 14; mo = 8; ml = 4; disp = 8;
                lit_in_code[i] = code_at[i] + 8;
            }
        } else {
            char prefix[PRM_PREFIX_MAX];
            size_t plen;
            if (pf[i].type == 5) {
                uint32_t b;
                memcpy(&b, &fv, 4);
                snprintf(prefix, sizeof prefix, "const float K_%s = 0x", names[i]);
                plen = strlen(prefix);
                snprintf(text + tl, 200, "%s%08x;\n", prefix, b);
                wlen = (uint16_t)(plen + 9); ml = 8;
            } else {
                snprintf(prefix, sizeof prefix, "const int K_%s = ", names[i]);
                plen = strlen(prefix);
                snprintf(text + tl, 200, "%s%-6d;\n", prefix, (int)pf[i].stored);
                wlen = (uint16_t)(plen + 7); ml = 6;
            }
            win_text_at[i] = tl;
            mo = (uint16_t)plen;
            r[68] = (uint8_t)plen;
            tl += strlen(text + tl);
        }
        w16(r + 32, wlen); w16(r + 34, (uint16_t)disp);
        r[43] = unmasked_fp ? 0 : 1;
        w16(r + 48, mo); w16(r + 50, ml);
    }

    /* ---- containers: the same text in two compressed entries (targets 1 and 2), an ignored entry, and a decoy header ---- */
    uint8_t *enc = malloc(tl + 64);
    size_t encn = lz4_encode_run((uint8_t *)text, tl, enc);
    for (int c = 0; c < 2; c++) {
        size_t total = 64 + encn + 64 + 16;
        size_t off = vec_alloc(&data, 16 + total, 8);
        uint8_t *h = data.d + off;
        w32(h, MAGIC); w16(h + 4, 1); w16(h + 6, 16); w64(h + 8, total);
        uint8_t *e = h + 16;
        w16(e, TEXT_KIND); w32(e + 4, 64); w64(e + 8, encn); w32(e + 16, (uint32_t)encn); w32(e + 28, (uint32_t)c + 1);
        w64(e + 40, BLOCK_FLAG); w64(e + 56, tl);
        memcpy(e + 64, enc, encn);
        uint8_t *e2 = e + 64 + encn;                            /* an entry of another kind: ignored */
        w16(e2, 3); w32(e2 + 4, 64); w64(e2 + 8, 16); w32(e2 + 28, 9);
    }
    free(enc);
    {
        size_t off = vec_alloc(&data, 32, 8);                   /* the magic with an impossible size */
        w32(data.d + off, MAGIC); w16(data.d + off + 4, 1); w16(data.d + off + 6, 16); w64(data.d + off + 8, 0xFFFFFFFFFFull);
    }
    vec_alloc(&data, 0x200, 1);

    /* ---- assemble the file ---- */
    size_t text_raw = 0x400, data_raw = text_raw + CODE_ROOM;
    size_t data_sz = data.n, total = data_raw + (data_sz + 0x1ff) / 0x200 * 0x200;
    uint8_t *lib = calloc(1, total);
    memcpy(lib + text_raw, code, CODE_ROOM);
    memcpy(lib + data_raw, data.d, data.n);
    lib[0] = 'M'; lib[1] = 'Z'; w32(lib + 60, 0x80);
    memcpy(lib + 0x80, "PE\0\0", 4);
    w16(lib + 0x84, 0x8664); w16(lib + 0x86, 2); w16(lib + 0x94, 0xF0); w16(lib + 0x98, 0x20b);
    uint8_t *sh = lib + 0x98 + 0xF0;
    uint32_t text_va = 0x1000, data_va = 0x1000 + ((CODE_ROOM + 0xfff) & ~0xfffu);
    memcpy(sh, ".text", 5);
    w32(sh + 8, CODE_ROOM); w32(sh + 12, text_va); w32(sh + 16, CODE_ROOM); w32(sh + 20, (uint32_t)text_raw); w32(sh + 36, 0x60000020);
    memcpy(sh + 40, ".rdata", 6);
    w32(sh + 48, (uint32_t)data_sz); w32(sh + 52, data_va); w32(sh + 56, (uint32_t)(total - data_raw)); w32(sh + 60, (uint32_t)data_raw);
    w32(sh + 76, 0x40000040);
    /* the relative references are patched now that the layout is known */
    for (int j = 0; j < NREC; j++) {
        int i = PIDX[j];
        if (!parrec[72 * j + 41]) continue;
        uint32_t ref_va = text_va + (uint32_t)code_at[i] + 8, next_va = ref_va + 4, target = data_va + (uint32_t)pool_at[i];
        w32(lib + text_raw + code_at[i] + 8, (uint32_t)(int32_t)(target - (next_va + 1)));
    }
    /* fingerprints of the (masked) windows, from the final bytes; literal offsets for the controls */
    for (int j = 0; j < NREC; j++) {
        int i = PIDX[j];
        uint8_t *r = parrec + 72 * j;
        uint16_t wlen = r16(r + 32);
        const uint8_t *src;
        if (pf[i].area == 0) { src = lib + data_raw + data_at[i]; W.param_lit_off[i] = data_raw + data_at[i] + 8; }
        else if (pf[i].area == 1) {
            src = lib + text_raw + code_at[i];
            W.param_lit_off[i] = r[41] ? data_raw + pool_at[i] : text_raw + lit_in_code[i];
        } else src = (const uint8_t *)text + win_text_at[i];
        uint8_t win[128] = {0};
        memcpy(win, src, wlen);
        memcpy(W.win8[j], win, 8);
        w32(r + 64, metalfg_fnv1a32(win, 8));       /* FNV-1a of the first eight window bytes, unmasked */
        if (!unmasked_fp) memset(win + r16(r + 48), 0, r16(r + 50));
        fgx_blake2b(r + 24, 8, win, wlen);
    }
    for (int i = 0; i < nb; i++) {
        W.blob_off[i] = data_raw + blob_rel[i];
        fgx_blake2b(blobrec + 40 * i, 8, lib + W.blob_off[i], 64);
    }

    /* ---- locator ---- */
    W.locsize = 104 + 40ull * 67 + 72ull * NREC;
    W.loc = calloc(1, W.locsize);
    memcpy(W.loc, "FGLOCAT3", 8);
    w32(W.loc + 8, 67); w32(W.loc + 12, NREC);
    metalfg_sha256(W.want_w + METALFG_WEIGHT_DATA_OFFSET, W.want_wn - METALFG_WEIGHT_DATA_OFFSET, W.loc + 16);
    metalfg_sha256(W.want_p, W.want_pn - 32, W.loc + 48);
    w32(W.loc + 80, MAGIC); w32(W.loc + 84, TEXT_KIND); w64(W.loc + 88, BLOCK_FLAG); w64(W.loc + 96, UNSUP_FLAG);
    memcpy(W.loc + 104, blobrec, sizeof blobrec);
    memcpy(W.loc + 104 + sizeof blobrec, parrec, sizeof parrec);
    W.lib = lib;
    W.libsize = total;
    free(data.d);
    for (int t = 0; t < METALFG_TENSOR_COUNT; t++) free(tens[t]);
    return W;
}
