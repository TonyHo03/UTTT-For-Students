package dk.easv.bll.bot;

import dk.easv.bll.game.GameManager;
import dk.easv.bll.game.IGameState;
import dk.easv.bll.move.IMove;
import dk.easv.bll.move.Move;

import java.util.Arrays;

/**
 *   - Bit-board state (O(1) win checks, zero array cloning per node)
 *   - Incremental Zobrist hashing (no full rehash per node)
 *   - Per-depth flat move buffers (zero ArrayList/boxing allocation)
 *   - Pure negamax α-β + PVS
 *   - Two-tier transposition table (persists across turns)
 *   - Killer + history heuristics
 *   - Enhanced eval: macro fork detection, destination danger, ownership
 *   - Opening book: center of center board
 */
public class NegaBot implements IBot {

    private static final String BOT_NAME  = "NegaBot";
    private static final double TIME_RATIO = 0.94;

    private static final int INF  = 10_000_000;
    private static final int WIN  =  1_000_000;
    private static final int DRAW = 0;

    private static final int TT_SIZE  = 1 << 20; // 1M buckets, two-tier = 2M slots
    private static final int TT_MASK  = TT_SIZE - 1;
    private static final int TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2;

    private static final int MAX_DEPTH    = 64;
    private static final int KILLER_SLOTS = 2;

    //  Bit-board line masks (precomputed at class load)
    // Each micro-board i occupies a contiguous 3×3 block in the 9×9 grid.
    // Cells are indexed as bit = x*9+y.  Bits 0-63 go in the low long, 64-80 in the high long.
    private static final long[][] MLINE_LO = new long[9][8]; // micro-line masks, low 64 bits
    private static final long[][] MLINE_HI = new long[9][8]; // micro-line masks, high 64 bits

    // 8 winning lines on the macro board (indices into the 9-cell macro array)
    private static final int[][] MAC_LINES = {
            {0,1,2},{3,4,5},{6,7,8},   // rows
            {0,3,6},{1,4,7},{2,5,8},   // cols
            {0,4,8},{2,4,6}            // diagonals
    };

    // Offsets within a micro-board for each of the 8 lines (in bit-space: row stride=9, col stride=1)
    private static final int[][] LINE_OFF = {
            {0,9,18},{1,10,19},{2,11,20},  // rows
            {0,1,2},{9,10,11},{18,19,20},  // cols
            {0,10,20},{2,10,18}            // diagonals
    };

    static {
        for (int mi = 0; mi < 9; mi++) {
            // base bit for top-left cell of micro-board mi
            int baseBit = (mi / 3) * 3 * 9 + (mi % 3) * 3;
            for (int li = 0; li < 8; li++) {
                long lo = 0, hi = 0;
                for (int off : LINE_OFF[li]) {
                    int b = baseBit + off;
                    if (b < 64) lo |= 1L << b;
                    else        hi |= 1L << (b - 64);
                }
                MLINE_LO[mi][li] = lo;
                MLINE_HI[mi][li] = hi;
            }
        }
    }

    //  Zobrist tables (static — deterministic, reused across games)
    private static final long[][] ZOB_CELL; // [81][2]  cell × player
    private static final long[][] ZOB_MAC;  // [9][5]   macro-cell state (0-4)
    private static final long     ZOB_SIDE;

    static {
        long s = 0xDEAD_BEEF_CAFE_1234L;
        ZOB_CELL = new long[81][2];
        for (int i = 0; i < 81; i++)
            for (int p = 0; p < 2; p++) { s = xs(s); ZOB_CELL[i][p] = s; }
        ZOB_MAC = new long[9][5];
        for (int i = 0; i < 9; i++)
            for (int v = 0; v < 5; v++) { s = xs(s); ZOB_MAC[i][v] = s; }
        s = xs(s); ZOB_SIDE = s;
    }

    private static long xs(long x) { x^=x<<13; x^=x>>>7; x^=x<<17; return x; }

    //  Transposition Table (static — persists across turns/games for free re-use)
    private static final long[] TT_KEY   = new long [TT_SIZE * 2];
    private static final int[]  TT_SCORE = new int  [TT_SIZE * 2];
    private static final byte[] TT_DEPTH = new byte [TT_SIZE * 2];
    private static final byte[] TT_TYPE  = new byte [TT_SIZE * 2];
    private static final byte[] TT_MX    = new byte [TT_SIZE * 2];
    private static final byte[] TT_MY    = new byte [TT_SIZE * 2];

    //  Bit-board game state
    /**
     * Compact game state using two pairs of longs for player bit-boards and a
     * single int for the macro-board (3 bits × 9 cells).
     *
     * Macro cell values: 0=open, 1=p0-won, 2=p1-won, 3=tie, 4=active
     */
    private static final class BS {
        long p0lo, p0hi; // player-0 pieces: bits 0-63 / 64-80
        long p1lo, p1hi; // player-1 pieces
        int  mac;        // packed macro state
        long hash;

        BS copy() {
            BS c = new BS();
            c.p0lo=p0lo; c.p0hi=p0hi; c.p1lo=p1lo; c.p1hi=p1hi;
            c.mac=mac; c.hash=hash;
            return c;
        }

        /** Is bit (x,y) set for player p? */
        boolean cell(int x, int y, int p) {
            int b = x*9+y;
            return p==0 ? (b<64 ? (p0lo>>>b&1)==1 : (p0hi>>>(b-64)&1)==1)
                    : (b<64 ? (p1lo>>>b&1)==1 : (p1hi>>>(b-64)&1)==1);
        }

        void setCell(int x, int y, int p) {
            int b = x*9+y;
            if (p==0) { if(b<64) p0lo|=1L<<b; else p0hi|=1L<<(b-64); }
            else      { if(b<64) p1lo|=1L<<b; else p1hi|=1L<<(b-64); }
        }

        /** Is cell (x,y) occupied by either player? */
        boolean occ(int x, int y) {
            int b = x*9+y;
            return b<64 ? ((p0lo|p1lo)>>>b&1)==1 : ((p0hi|p1hi)>>>(b-64)&1)==1;
        }

        int  ms(int mi)           { return (mac>>>(mi*3))&7; }
        void setMs(int mi, int v) { mac = (mac & ~(7<<(mi*3))) | (v<<(mi*3)); }
    }

    //  Per-search state

    private int     myPlayer;
    private long    t0, tLimit;
    private boolean timedOut;

    private final int[][] killers = new int[MAX_DEPTH][KILLER_SLOTS];
    private final int[][] hist    = new int[9][9]; // [macro-cell][local-cell]

    // Per-depth flat move buffers — avoids ALL ArrayList allocation during search
    private final int[][] MX = new int[MAX_DEPTH][81];
    private final int[][] MY = new int[MAX_DEPTH][81];
    private final int[][] OS = new int[MAX_DEPTH][81]; // ordering scores

    @Override
    public IMove doMove(IGameState state) {
        myPlayer = state.getMoveNumber() % 2;
        tLimit   = (long)(state.getTimePerMove() * TIME_RATIO);
        t0       = System.currentTimeMillis();
        timedOut = false;

        for (int[] r : killers) Arrays.fill(r, -1);
        for (int[] r : hist)    Arrays.fill(r, 0);

        BS root = encode(state);

        // Opening book: always take center-of-center if available
        if (!root.occ(4,4) && root.ms(4)==4) return new Move(4,4);

        int nm = gen(root, 0);
        if (nm == 0) return null;
        if (nm == 1) return new Move(MX[0][0], MY[0][0]);

        IMove best = new Move(MX[0][0], MY[0][0]);

        // Iterative deepening — 100% of budget on exact negamax
        for (int d = 1; d < MAX_DEPTH; d++) {
            if (timeout()) break;
            IMove cand = rootSearch(root, d);
            if (!timedOut) {
                best = cand;
                if (decisive(ttPeek(root.hash, d))) break; // forced win/loss
            } else break;
        }

        return best;
    }

    //  Root search (PVS, tracks best move explicitly)
    private IMove rootSearch(BS s, int depth) {
        int nm = gen(s, 0);
        order(s, nm, ttMove(s.hash), 0);

        IMove bestM = new Move(MX[0][0], MY[0][0]);
        int best = -INF, a = -INF, b = INF;
        boolean first = true;

        for (int i = 0; i < nm; i++) {
            if (timeout()) { timedOut = true; break; }
            int x = MX[0][i], y = MY[0][i];
            AR ar = apply(s, x, y, myPlayer);
            int sc;
            if      (ar.over == GameManager.GameOverState.Win) sc = WIN + depth;
            else if (ar.over == GameManager.GameOverState.Tie) sc = DRAW;
            else if (first)  sc = -negamax(ar.ns, depth-1, -b, -a, 1);
            else {
                sc = -negamax(ar.ns, depth-1, -a-1, -a, 1);
                if (!timedOut && sc > a && sc < b)
                    sc = -negamax(ar.ns, depth-1, -b, -a, 1);
            }
            if (!timedOut && sc > best) { best = sc; bestM = new Move(x, y); }
            if (sc > a) a = sc;
            if (a >= b) break;
            first = false;
        }
        if (!timedOut) ttStore(s.hash, depth, best, TT_EXACT, bestM);
        return bestM;
    }

    //  Negamax AlphaBeta pruning + PVS
    private int negamax(BS s, int depth, int a, int b, int ply) {
        if (timeout()) { timedOut = true; return 0; }
        int a0 = a;

        int[] hit = ttProbe(s.hash, depth, a, b);
        if (hit != null) return hit[0];
        IMove ttm = ttMove(s.hash);

        int cp = (myPlayer + ply) % 2;
        int nm = gen(s, ply);
        if (nm == 0 || depth <= 0) {
            int ev = eval(s);
            return cp == myPlayer ? ev : -ev;
        }

        order(s, nm, ttm, ply);
        int best = -INF;
        IMove bestM = new Move(MX[ply][0], MY[ply][0]);
        boolean first = true;

        for (int i = 0; i < nm; i++) {
            if (timeout()) { timedOut = true; return best; }
            int x = MX[ply][i], y = MY[ply][i];
            AR ar = apply(s, x, y, cp);
            int sc;
            if (ar.over == GameManager.GameOverState.Win) {
                sc = WIN + depth;
            } else if (ar.over == GameManager.GameOverState.Tie) {
                sc = DRAW;
            } else if (first) {
                sc = -negamax(ar.ns, depth-1, -b, -a, ply+1);
            } else {
                sc = -negamax(ar.ns, depth-1, -a-1, -a, ply+1);
                if (!timedOut && sc > a && sc < b)
                    sc = -negamax(ar.ns, depth-1, -b, -a, ply+1);
            }
            if (sc > best) { best = sc; bestM = new Move(x, y); }
            if (sc > a) a = sc;
            if (a >= b) {
                killUp(x*9+y, ply);
                histUp(x, y, depth);
                break;
            }
            first = false;
        }

        if (!timedOut) {
            int t = best <= a0 ? TT_UPPER : best >= b ? TT_LOWER : TT_EXACT;
            ttStore(s.hash, depth, best, t, bestM);
        }
        return best;
    }

    //  Move application — returns new BS (copy-on-write via BS.copy())
    private static final class AR {
        final BS ns;
        final GameManager.GameOverState over;
        AR(BS n, GameManager.GameOverState o) { ns=n; over=o; }
    }

    private AR apply(BS s, int x, int y, int player) {
        BS ns = s.copy();
        int mx = x/3, my = y/3, mi = mx*3+my;
        ns.setCell(x, y, player);
        ns.hash ^= ZOB_CELL[x*9+y][player];

        GameManager.GameOverState over = GameManager.GameOverState.Active;

        if (microWin(ns, mx, my, player)) {
            ns.hash ^= ZOB_MAC[mi][ns.ms(mi)];
            ns.setMs(mi, player+1);
            ns.hash ^= ZOB_MAC[mi][player+1];
            if (macroWin(ns, player)) over = GameManager.GameOverState.Win;
        } else if (microFull(ns, mx, my)) {
            ns.hash ^= ZOB_MAC[mi][ns.ms(mi)];
            ns.setMs(mi, 3);
            ns.hash ^= ZOB_MAC[mi][3];
            if (macroFull(ns)) over = GameManager.GameOverState.Tie;
        }

        // Clear all currently-active boards
        for (int i = 0; i < 9; i++) {
            if (ns.ms(i) == 4) {
                ns.hash ^= ZOB_MAC[i][4]; ns.setMs(i, 0); ns.hash ^= ZOB_MAC[i][0];
            }
        }
        // Activate target board (or all open boards if target is closed)
        int tx = x%3, ty = y%3, tgt = tx*3+ty;
        if (ns.ms(tgt) == 0) {
            ns.hash ^= ZOB_MAC[tgt][0]; ns.setMs(tgt, 4); ns.hash ^= ZOB_MAC[tgt][4];
        } else {
            for (int i = 0; i < 9; i++) {
                if (ns.ms(i) == 0) {
                    ns.hash ^= ZOB_MAC[i][0]; ns.setMs(i, 4); ns.hash ^= ZOB_MAC[i][4];
                }
            }
        }
        ns.hash ^= ZOB_SIDE;
        return new AR(ns, over);
    }

    //  State encoding from IGameState
    private BS encode(IGameState gs) {
        BS s = new BS();
        String[][] b  = gs.getField().getBoard();
        String[][] mb = gs.getField().getMacroboard();
        for (int x = 0; x < 9; x++) for (int y = 0; y < 9; y++) {
            if      (b[x][y].equals("0")) s.setCell(x, y, 0);
            else if (b[x][y].equals("1")) s.setCell(x, y, 1);
        }
        for (int mx = 0; mx < 3; mx++) for (int my = 0; my < 3; my++) {
            int mi = mx*3+my;
            int v;
            switch (mb[mx][my]) {
                case "0":   v = 1; break;
                case "1":   v = 2; break;
                case "TIE": v = 3; break;
                case "-1":  v = 4; break;
                default:    v = 0; break;
            }
            s.setMs(mi, v);
        }
        s.hash = fullHash(s, myPlayer);
        return s;
    }

    //  Win / tie detection (bit-board O(1))
    private boolean microWin(BS s, int mx, int my, int p) {
        int mi = mx*3+my;
        long plo = p==0 ? s.p0lo : s.p1lo;
        long phi = p==0 ? s.p0hi : s.p1hi;
        for (int li = 0; li < 8; li++) {
            long ml = MLINE_LO[mi][li], mh = MLINE_HI[mi][li];
            if ((plo & ml)==ml && (phi & mh)==mh) return true;
        }
        return false;
    }

    private boolean macroWin(BS s, int p) {
        int pv = p+1;
        for (int[] ln : MAC_LINES)
            if (s.ms(ln[0])==pv && s.ms(ln[1])==pv && s.ms(ln[2])==pv) return true;
        return false;
    }

    private boolean microFull(BS s, int mx, int my) {
        for (int lx = 0; lx < 3; lx++) for (int ly = 0; ly < 3; ly++)
            if (!s.occ(mx*3+lx, my*3+ly)) return false;
        return true;
    }

    private boolean macroFull(BS s) {
        for (int i = 0; i < 9; i++) { int v = s.ms(i); if (v==0||v==4) return false; }
        return true;
    }

    /** Would player p win micro-board by playing at (x,y)? (Used in ordering only) */
    private boolean winsMicro(BS s, int x, int y, int p) {
        // Temporarily set the bit without copying the full state
        int b = x*9+y;
        if (p == 0) {
            if (b < 64) { s.p0lo |= 1L<<b; boolean r = microWin(s,x/3,y/3,0); s.p0lo &= ~(1L<<b); return r; }
            else { s.p0hi |= 1L<<(b-64); boolean r = microWin(s,x/3,y/3,0); s.p0hi &= ~(1L<<(b-64)); return r; }
        } else {
            if (b < 64) { s.p1lo |= 1L<<b; boolean r = microWin(s,x/3,y/3,1); s.p1lo &= ~(1L<<b); return r; }
            else { s.p1hi |= 1L<<(b-64); boolean r = microWin(s,x/3,y/3,1); s.p1hi &= ~(1L<<(b-64)); return r; }
        }
    }

    //  Move generation (into flat per-depth buffers — zero allocation)
    private int gen(BS s, int ply) {
        int n = 0;
        for (int macX = 0; macX < 3; macX++) for (int macY = 0; macY < 3; macY++) {
            if (s.ms(macX*3+macY) != 4) continue;
            for (int lx = 0; lx < 3; lx++) for (int ly = 0; ly < 3; ly++) {
                int x = macX*3+lx, y = macY*3+ly;
                if (!s.occ(x, y)) { MX[ply][n] = x; MY[ply][n] = y; n++; }
            }
        }
        return n;
    }

    //  Move ordering (insertion sort on flat arrays — zero allocation)
    private void order(BS s, int nm, IMove ttm, int ply) {
        int cp  = (myPlayer + ply) % 2;
        int pv  = cp + 1;          // my value (1 or 2)
        int ov  = (cp^1) + 1;      // opponent value
        for (int i = 0; i < nm; i++)
            OS[ply][i] = scoreMove(s, MX[ply][i], MY[ply][i], cp, pv, ov, ttm, ply);
        isort(nm, ply);
    }

    private int scoreMove(BS s, int x, int y, int cp, int pv, int ov, IMove ttm, int ply) {
        int sc = 0;
        int mx = x/3, my = y/3, lx = x%3, ly = y%3;

        // TT / PV move — search this first
        if (ttm != null && ttm.getX()==x && ttm.getY()==y) sc += 2_000_000;

        // Immediate micro-win
        boolean iWin  = winsMicro(s, x, y, cp);
        boolean oppWin = winsMicro(s, x, y, cp^1);
        if (iWin)  sc += 1_000_000;
        if (oppWin) sc +=   500_000;  // blocking opponent win

        // Macro fork: how many macro threats does winning this board create?
        if (iWin) sc += 100_000 * macroThreatsAfter(s, mx, my, cp);

        // Killers
        if (ply < MAX_DEPTH) {
            int enc = x*9+y;
            if (killers[ply][0] == enc) sc += 90_000;
            else if (killers[ply][1] == enc) sc += 80_000;
        }

        // History heuristic
        sc += Math.min(hist[mx*3+my][lx*3+ly], 70_000);

        // Local position
        if (lx==1 && ly==1) sc += 300;
        else if ((lx&1)==0 && (ly&1)==0) sc += 150;
        else sc += 50;
        if (mx==1 && my==1) sc += 200;
        else if ((mx&1)==0 && (my&1)==0) sc += 100;

        // Destination danger: penalise sending opponent to a board where they have an immediate win
        int tx = x%3, ty = y%3, tst = s.ms(tx*3+ty);
        if (tst==0 || tst==4) {
            if (oppHasImmWin(s, tx, ty, cp^1)) sc -= 200_000;
            if (tx==1 && ty==1) sc -= 150; // sending to center is especially dangerous
        } else {
            sc += 200; // sending to closed board = free choice for us next
        }

        return sc;
    }

    private int macroThreatsAfter(BS s, int mx, int my, int player) {
        int pv = player+1, mi = mx*3+my, count = 0;
        for (int[] ln : MAC_LINES) {
            int mine = 0, free = 0;
            for (int idx : ln) {
                int v = (idx == mi) ? pv : s.ms(idx);
                if (v == pv) mine++; else if (v==0||v==4) free++;
            }
            if (mine == 2 && free >= 1) count++;
        }
        return count;
    }

    private boolean oppHasImmWin(BS s, int tx, int ty, int opp) {
        for (int lx = 0; lx < 3; lx++) for (int ly = 0; ly < 3; ly++) {
            int x = tx*3+lx, y = ty*3+ly;
            if (!s.occ(x,y) && winsMicro(s, x, y, opp)) return true;
        }
        return false;
    }

    private void isort(int n, int ply) {
        for (int i = 1; i < n; i++) {
            int ks = OS[ply][i], kx = MX[ply][i], ky = MY[ply][i], j = i-1;
            while (j >= 0 && OS[ply][j] < ks) {
                OS[ply][j+1] = OS[ply][j]; MX[ply][j+1] = MX[ply][j]; MY[ply][j+1] = MY[ply][j]; j--;
            }
            OS[ply][j+1] = ks; MX[ply][j+1] = kx; MY[ply][j+1] = ky;
        }
    }

    //  Killer & History
    private void killUp(int enc, int ply) {
        if (ply >= MAX_DEPTH) return;
        if (killers[ply][0] != enc) { killers[ply][1] = killers[ply][0]; killers[ply][0] = enc; }
    }

    private void histUp(int x, int y, int d) {
        hist[x/3*3+y/3][x%3*3+y%3] += d*d;
    }

    //  Static evaluation (from myPlayer's absolute perspective)
    private int eval(BS s) {
        int me = myPlayer+1, op = (myPlayer^1)+1;
        int sc = macroEval(s, me, op);
        for (int mx = 0; mx < 3; mx++) for (int my = 0; my < 3; my++) {
            int st = s.ms(mx*3+my);
            if (st==1||st==2||st==3) continue; // decided board — skip micro eval
            sc += microEval(s, mx, my, me, op) * mw(mx, my);
        }
        return sc;
    }

    private int macroEval(BS s, int me, int op) {
        int sc = 0;
        // Open-line scoring on the 3×3 macro grid
        for (int[] ln : MAC_LINES) {
            int mc = 0, oc = 0;
            for (int i : ln) {
                int v = s.ms(i);
                if (v==me) mc++; else if (v==op) oc++; else if (v==3) { mc=oc=3; break; }
            }
            if (oc==0) { if (mc==2) sc+=1000; else if (mc==1) sc+=200; }
            if (mc==0) { if (oc==2) sc-=1000; else if (oc==1) sc-=200; }
        }
        // Won/owned macro cells
        for (int i = 0; i < 9; i++) {
            int w = mw(i/3, i%3), v = s.ms(i);
            if (v==me) sc += 500 + w*60;
            if (v==op) sc -= 500 + w*60;
        }
        // Fork bonus: having ≥2 simultaneous macro threats is decisive
        int mt = macroThreats(s, me), ot = macroThreats(s, op);
        if (mt >= 2) sc += 350 * mt;
        if (ot >= 2) sc -= 450 * ot;
        return sc;
    }

    private int macroThreats(BS s, int pv) {
        int c = 0;
        for (int[] ln : MAC_LINES) {
            int mine = 0, free = 0, bl = 0;
            for (int i : ln) {
                int v = s.ms(i);
                if (v==pv) mine++; else if (v==0||v==4) free++; else bl++;
            }
            if (mine==2 && free>=1 && bl==0) c++;
        }
        return c;
    }

    private int microEval(BS s, int mx, int my, int me, int op) {
        int mi = mx*3+my, sc = 0;
        long mlo = me==1 ? s.p0lo : s.p1lo, mhi = me==1 ? s.p0hi : s.p1hi;
        long olo = op==1 ? s.p0lo : s.p1lo, ohi = op==1 ? s.p0hi : s.p1hi;
        for (int li = 0; li < 8; li++) {
            long ll = MLINE_LO[mi][li], lh = MLINE_HI[mi][li];
            int mc = Long.bitCount(mlo&ll) + Long.bitCount(mhi&lh);
            int oc = Long.bitCount(olo&ll) + Long.bitCount(ohi&lh);
            if (oc==0) { if (mc==2) sc+=25; else if (mc==1) sc+=7; }
            if (mc==0) { if (oc==2) sc-=25; else if (oc==1) sc-=7; }
        }
        // Center and corner bonuses within micro-board
        int cx = mx*3+1, cy = my*3+1;
        if (s.cell(cx, cy, me-1)) sc += 10;
        if (s.cell(cx, cy, op-1)) sc -= 10;
        int[][] corners = {{0,0},{0,2},{2,0},{2,2}};
        for (int[] c : corners) {
            if (s.cell(mx*3+c[0], my*3+c[1], me-1)) sc += 4;
            if (s.cell(mx*3+c[0], my*3+c[1], op-1)) sc -= 4;
        }
        return sc;
    }

    private int mw(int mx, int my) {
        if (mx==1 && my==1) return 4;
        if ((mx&1)==0 && (my&1)==0) return 3;
        return 2;
    }

    //  Zobrist hashing
    private long fullHash(BS s, int side) {
        long h = 0;
        for (int x = 0; x < 9; x++) for (int y = 0; y < 9; y++) {
            if (s.cell(x,y,0)) h ^= ZOB_CELL[x*9+y][0];
            if (s.cell(x,y,1)) h ^= ZOB_CELL[x*9+y][1];
        }
        for (int i = 0; i < 9; i++) { int v = s.ms(i); if (v>0) h ^= ZOB_MAC[i][v]; }
        if (side == 1) h ^= ZOB_SIDE;
        return h;
    }

    //  Transposition Table
    private void ttStore(long h, int d, int sc, int t, IMove m) {
        int base = (int)(h & TT_MASK) * 2;
        // Depth-preferred slot
        if (TT_KEY[base]==0 || d >= (TT_DEPTH[base]&0xFF)) {
            TT_KEY[base]=h; TT_SCORE[base]=sc; TT_DEPTH[base]=(byte)Math.min(d,127);
            TT_TYPE[base]=(byte)t;
            TT_MX[base]=(byte)(m!=null ? m.getX() : -1);
            TT_MY[base]=(byte)(m!=null ? m.getY() : -1);
        }
        // Always-replace slot
        int s1 = base+1;
        TT_KEY[s1]=h; TT_SCORE[s1]=sc; TT_DEPTH[s1]=(byte)Math.min(d,127);
        TT_TYPE[s1]=(byte)t;
        TT_MX[s1]=(byte)(m!=null ? m.getX() : -1);
        TT_MY[s1]=(byte)(m!=null ? m.getY() : -1);
    }

    private int[] ttProbe(long h, int d, int a, int b) {
        int base = (int)(h & TT_MASK) * 2;
        for (int sl = base; sl <= base+1; sl++) {
            if (TT_KEY[sl]!=h || (TT_DEPTH[sl]&0xFF)<d) continue;
            int sc = TT_SCORE[sl], t = TT_TYPE[sl];
            if (t==TT_EXACT) return new int[]{sc};
            if (t==TT_LOWER && sc>=b) return new int[]{sc};
            if (t==TT_UPPER && sc<=a) return new int[]{sc};
        }
        return null;
    }

    private int ttPeek(long h, int d) {
        int base = (int)(h & TT_MASK) * 2;
        for (int sl = base; sl <= base+1; sl++)
            if (TT_KEY[sl]==h && (TT_DEPTH[sl]&0xFF)>=d && TT_TYPE[sl]==TT_EXACT)
                return TT_SCORE[sl];
        return 0;
    }

    private IMove ttMove(long h) {
        int base = (int)(h & TT_MASK) * 2;
        for (int sl = base; sl <= base+1; sl++)
            if (TT_KEY[sl]==h && TT_MX[sl]>=0) return new Move(TT_MX[sl], TT_MY[sl]);
        return null;
    }

    private boolean decisive(int sc) { return Math.abs(sc) >= WIN/2; }
    private boolean timeout()        { return ms() >= tLimit; }
    private long    ms()             { return System.currentTimeMillis() - t0; }

    @Override
    public String getBotName() { return BOT_NAME; }
}