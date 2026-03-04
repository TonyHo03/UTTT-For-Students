package dk.easv.bll.bot;

import dk.easv.bll.game.GameManager;
import dk.easv.bll.game.IGameState;
import dk.easv.bll.move.IMove;
import dk.easv.bll.move.Move;

import java.util.Arrays;
import java.util.List;

public class NegaMaxBot implements IBot {

    //  Constants

    private static final String BOT_NAME   = "NegaMaxBot";

    /** How much of the allowed time we actually use (leaves a safety margin). */
    private static final double TIME_RATIO = 0.95;

    /** Terminal / near-terminal scores. */
    private static final int INF      = 10_000_000;
    private static final int WIN      =  1_000_000;
    private static final int DRAW     =  0;

    /** Transposition table size – must be a power of 2. */
    private static final int TT_SIZE  = 1 << 20;   // 1 048 576 entries (~two-tier → 2M slots)
    private static final int TT_MASK  = TT_SIZE - 1;

    private static final int TT_EXACT = 0;
    private static final int TT_LOWER = 1; // fail-high (alpha bound)
    private static final int TT_UPPER = 2; // fail-low (beta  bound)

    /** Killer heuristic: 2 killers per ply, up to depth 64. */
    private static final int MAX_DEPTH   = 64;
    private static final int KILLER_SLOTS = 2;

    //  Zobrist hash tables  (initialised once, static → shared across games)

    /**
     * zobrist[cell][player]
     *   cell   = x*9 + y   (0..80 for the 9×9 board)
     *   player = 0 or 1
     * Extra entries for macro-board state and side-to-move.
     */
    private static final long[][] ZOB_BOARD; // [81][2]
    private static final long[][] ZOB_MACRO; // [9][4] (p0, p1, tie, available)
    private static final long ZOB_SIDE;

    static {
        // Deterministic PRNG so the hash table is reproducible across runs.
        long seed = 0xDEAD_BEEF_CAFE_1234L;
        ZOB_BOARD = new long[81][2];
        for (int i = 0; i < 81; i++)
            for (int p = 0; p < 2; p++) {
                seed = xorShift64(seed);
                ZOB_BOARD[i][p] = seed;
            }
        ZOB_MACRO = new long[9][4];
        for (int i = 0; i < 9; i++)
            for (int s = 0; s < 4; s++) {
                seed = xorShift64(seed);
                ZOB_MACRO[i][s] = seed;
            }
        seed = xorShift64(seed);
        ZOB_SIDE = seed;
    }

    private static long xorShift64(long x) {
        x ^= x << 13;
        x ^= x >>> 7;
        x ^= x << 17;
        return x;
    }

    //  Transposition Table

    /** Two-tier table: slot 0 = depth-preferred, slot 1 = always-replace. */
    private static final long[] TT_KEY = new long [TT_SIZE * 2];
    private static final int[] TT_SCORE = new int [TT_SIZE * 2];
    private static final byte[] TT_DEPTH = new byte [TT_SIZE * 2];
    private static final byte[] TT_TYPE = new byte [TT_SIZE * 2];
    private static final byte[] TT_MOVEX = new byte [TT_SIZE * 2];
    private static final byte[] TT_MOVEY = new byte [TT_SIZE * 2];

    //  Per-search state

    private int  myPlayer;   // 0 or 1
    private long startTime;
    private long timeLimitMs;
    private boolean timedOut;

    /** Killer moves [ply][slot] encoded as x*9+y. */
    private final int[][] killers = new int[MAX_DEPTH][KILLER_SLOTS];

    /** History heuristic [from-cell][to-cell] – 'from' = macro cell (0-8). */
    private final int[][] history = new int[81][81];

    //  IBot entry point

    @Override
    public IMove doMove(IGameState state) {
        myPlayer = state.getMoveNumber() % 2; // 0 or 1
        timeLimitMs = (long)(state.getTimePerMove() * TIME_RATIO);
        startTime = System.currentTimeMillis();
        timedOut = false;

        // Reset per-search tables (history decays, killers reset)
        for (int[] row : killers) Arrays.fill(row,-1);
        for (int[] row : history) Arrays.fill(row,0);

        List<IMove> moves = state.getField().getAvailableMoves();
        if (moves.isEmpty()) return null;
        if (moves.size() == 1) return moves.get(0);

        // Build compact state
        int[] board = encodeBoard(state);
        int[] macro = encodeMacro(state);
        long hash = computeHash(board, macro, myPlayer);

        IMove bestMove = moves.get(0);

        // Iterative Deepening
        for (int depth = 1; depth < MAX_DEPTH; depth++) {
            IMove candidate = searchRoot(board, macro, hash, depth);
            if (!timedOut) {
                bestMove = candidate;
                // Forced win found – no point searching deeper
                if (isDecisiveScore(ttProbeScore(hash, depth))) break;
            } else {
                break;
            }
        }
        return bestMove;
    }

    //  Root search (PVS at the root so we can track the best move)

    private IMove searchRoot(int[] board, int[] macro, long hash, int depth) {
        List<IMove> moves = generateMoves(board, macro);
        orderMovesRoot(moves, board, macro, hash);

        IMove bestMove = moves.get(0);
        int bestScore = -INF;
        int alpha = -INF;
        int beta = INF;
        boolean firstMove = true;

        for (int idx = 0; idx < moves.size(); idx++) {
            if (timeout()) { timedOut = true; break; }

            IMove move = moves.get(idx);
            MoveResult mr = applyMove(board, macro, move, myPlayer);

            int score;
            if (mr.gameOver == GameManager.GameOverState.Win) {
                score = WIN + depth;
            } else if (mr.gameOver == GameManager.GameOverState.Tie) {
                score = DRAW;
            } else if (firstMove) {
                score = -negamax(mr.board, mr.macro, mr.hash, depth - 1, -beta, -alpha, 1);
            } else {
                // Null-window search
                score = -negamax(mr.board, mr.macro, mr.hash, depth - 1, -alpha - 1, -alpha, 1);
                // Re-search if it failed high
                if (!timedOut && score > alpha && score < beta) {
                    score = -negamax(mr.board, mr.macro, mr.hash, depth - 1, -beta, -alpha, 1);
                }
            }

            if (!timedOut && score > bestScore) {
                bestScore = score;
                bestMove  = move;
            }
            if (score > alpha) alpha = score;
            if (alpha >= beta) break;
            firstMove = false;
        }

        // Store root result in TT
        if (!timedOut) {
            ttStore(hash, depth, bestScore, TT_EXACT, bestMove);
        }

        return bestMove;
    }

    //  Negamax with PVS + Alpha-Beta + TT

    private int negamax(int[] board, int[] macro, long hash,
                        int depth, int alpha, int beta, int ply) {
        if (timeout()) { timedOut = true; return 0; }

        int alphaOrig = alpha;

        // ── Transposition Table Probe ────────────────────────────────────────
        int[] ttResult = ttProbe(hash, depth, alpha, beta);
        if (ttResult != null) return ttResult[0];
        IMove ttMove = ttProbeMove(hash);

        // ── Terminal / leaf nodes ────────────────────────────────────────────
        int currentPlayer = (myPlayer + ply) % 2; // side to move at this ply
        List<IMove> moves = generateMoves(board, macro);
        if (moves.isEmpty() || depth <= 0) {
            // Evaluate from myPlayer's absolute view, then flip if opponent is to move
            int eval = staticEval(board, macro);
            return (currentPlayer == myPlayer) ? eval : -eval;
        }

        // ── Move ordering ────────────────────────────────────────────────────
        orderMoves(moves, board, macro, ttMove, ply);
        int bestScore = -INF;
        IMove bestMove = moves.get(0);
        boolean firstMove = true;

        for (IMove move : moves) {
            if (timeout()) { timedOut = true; return bestScore; }

            MoveResult mr = applyMove(board, macro, move, currentPlayer);

            int score;
            if (mr.gameOver == GameManager.GameOverState.Win) {
                score = WIN + depth;
            } else if (mr.gameOver == GameManager.GameOverState.Tie) {
                score = DRAW;
            } else if (firstMove) {
                // Full-window search on first (presumed best) move
                score = -negamax(mr.board, mr.macro, mr.hash, depth - 1, -beta, -alpha, ply + 1);
            } else {
                // ── Principal Variation Search (null-window) ─────────────────
                score = -negamax(mr.board, mr.macro, mr.hash, depth - 1, -alpha - 1, -alpha, ply + 1);
                if (!timedOut && score > alpha && score < beta) {
                    // Fail-high: re-search with full window
                    score = -negamax(mr.board, mr.macro, mr.hash, depth - 1, -beta, -alpha, ply + 1);
                }
            }

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
            if (score > alpha) alpha = score;
            if (alpha >= beta) {
                // ── Killer + History update ──────────────────────────────────
                updateKillers(move, ply);
                updateHistory(move, depth);
                break; // Beta cut-off
            }
            firstMove = false;
        }

        // ── TT store ─────────────────────────────────────────────────────────
        if (!timedOut) {
            int ttType = (bestScore <= alphaOrig) ? TT_UPPER
                    : (bestScore >= beta) ? TT_LOWER
                    : TT_EXACT;
            ttStore(hash, depth, bestScore, ttType, bestMove);
        }

        return bestScore;
    }

    // =========================================================================
    //  Board representation  (compact int arrays for speed)
    // =========================================================================

    /**
     * board[x*9+y] encoding:
     *   0 = empty / available  (EMPTY_FIELD or AVAILABLE_FIELD)
     *   1 = player 0
     *   2 = player 1
     */
    private int[] encodeBoard(IGameState state) {
        String[][] raw = state.getField().getBoard();
        int[] b = new int[81];
        for (int x = 0; x < 9; x++)
            for (int y = 0; y < 9; y++) {
                String v = raw[x][y];
                if (v.equals("0")) b[x * 9 + y] = 1;
                else if (v.equals("1")) b[x * 9 + y] = 2;
                else b[x * 9 + y] = 0;
            }
        return b;
    }

    /**
     * macro[mx*3+my] encoding:
     *   0 = empty (not yet won, not active)
     *   1 = player 0 won
     *   2 = player 1 won
     *   3 = tie
     *   4 = active (AVAILABLE_FIELD, "-1")
     */
    private int[] encodeMacro(IGameState state) {
        String[][] raw = state.getField().getMacroboard();
        int[] m = new int[9];
        for (int mx = 0; mx < 3; mx++)
            for (int my = 0; my < 3; my++) {
                String v = raw[mx][my];
                if (v.equals("0"))  m[mx * 3 + my] = 1;
                else if (v.equals("1")) m[mx * 3 + my] = 2;
                else if (v.equals("TIE"))m[mx * 3 + my] = 3;
                else if (v.equals("-1")) m[mx * 3 + my] = 4;
                else m[mx * 3 + my] = 0;
            }
        return m;
    }

    // =========================================================================
    //  Move generation
    // =========================================================================

    private List<IMove> generateMoves(int[] board, int[] macro) {
        java.util.ArrayList<IMove> list = new java.util.ArrayList<>(20);
        for (int mx = 0; mx < 3; mx++) {
            for (int my = 0; my < 3; my++) {
                if (macro[mx * 3 + my] != 4) continue;  // only active boards
                for (int lx = 0; lx < 3; lx++) {
                    for (int ly = 0; ly < 3; ly++) {
                        int x = mx * 3 + lx;
                        int y = my * 3 + ly;
                        if (board[x * 9 + y] == 0) {
                            list.add(new Move(x, y));
                        }
                    }
                }
            }
        }
        return list;
    }

    // =========================================================================
    //  Move application  (no object allocation for the board – in-place + undo)
    //  Returns a MoveResult with NEW arrays (needed because we can't easily undo
    //  macro changes; board undo is trivial but macro undo is complex).
    //  We minimise allocation by reusing a pool… but for safety just clone here.
    // =========================================================================

    private static class MoveResult {
        final int[] board;
        final int[] macro;
        final long hash;
        final GameManager.GameOverState gameOver;
        MoveResult(int[] b, int[] m, long h, GameManager.GameOverState g) {
            board = b; macro = m; hash = h; gameOver = g;
        }
    }

    private MoveResult applyMove(int[] board, int[] macro, IMove move, int player) {
        int[] nb = board.clone();
        int[] nm = macro.clone();

        int x = move.getX(), y = move.getY();
        int cell = x * 9 + y;
        int pval = player + 1;   // 1 = p0, 2 = p1

        nb[cell] = pval;

        int mx = x / 3, my = y / 3;
        int mCell = mx * 3 + my;

        GameManager.GameOverState gameOver = GameManager.GameOverState.Active;

        // Check if micro-board is won or tied
        if (nm[mCell] == 4 || nm[mCell] == 0) {
            if (checkMicroWin(nb, mx, my, pval)) {
                nm[mCell] = pval;
                // Check macro win
                if (checkMacroWin(nm, mx, my, pval)) {
                    gameOver = GameManager.GameOverState.Win;
                }
            } else if (checkMicroFull(nb, mx, my)) {
                nm[mCell] = 3;  // tie
                if (checkMacroFull(nm)) {
                    gameOver = GameManager.GameOverState.Tie;
                }
            }
        }

        // Update active boards
        for (int i = 0; i < 9; i++)
            if (nm[i] == 4) nm[i] = 0;

        int tx = x % 3, ty = y % 3;
        int targetMacro = tx * 3 + ty;
        if (nm[targetMacro] == 0) {
            nm[targetMacro] = 4;
        } else {
            // Target already decided – open all undecided boards
            for (int i = 0; i < 9; i++)
                if (nm[i] == 0) nm[i] = 4;
        }

        // Hash computed after full state update; next player to move is opponent
        long hash = computeHash(nb, nm, (player + 1) % 2);

        return new MoveResult(nb, nm, hash, gameOver);
    }

    // =========================================================================
    //  Win / tie checks  (fast, no String comparisons)
    // =========================================================================

    /** Check if player pval just won the micro-board at (mx, my). */
    private boolean checkMicroWin(int[] board, int mx, int my, int pval) {
        int sx = mx * 3, sy = my * 3;
        // rows
        for (int lx = 0; lx < 3; lx++)
            if (board[(sx+lx)*9+sy]==pval && board[(sx+lx)*9+sy+1]==pval && board[(sx+lx)*9+sy+2]==pval) return true;
        // cols
        for (int ly = 0; ly < 3; ly++)
            if (board[sx*9+sy+ly]==pval && board[(sx+1)*9+sy+ly]==pval && board[(sx+2)*9+sy+ly]==pval) return true;
        // diags
        if (board[sx*9+sy]==pval && board[(sx+1)*9+sy+1]==pval && board[(sx+2)*9+sy+2]==pval)
            return true;
        if (board[sx*9+sy+2]==pval && board[(sx+1)*9+sy+1]==pval && board[(sx+2)*9+sy]==pval)
            return true;
        return false;
    }

    /** Check if the macro board is won by pval after placing at (mx,my). */
    private boolean checkMacroWin(int[] macro, int mx, int my, int pval) {
        // reuse same pattern with macro indices 0..8
        int[] m = macro;
        // rows
        for (int r = 0; r < 3; r++)
            if (m[r*3]==pval && m[r*3+1]==pval && m[r*3+2]==pval) return true;
        // cols
        for (int c = 0; c < 3; c++)
            if (m[c]==pval && m[3+c]==pval && m[6+c]==pval) return true;
        // diags
        if (m[0]==pval && m[4]==pval && m[8]==pval) return true;
        if (m[2]==pval && m[4]==pval && m[6]==pval) return true;
        return false;
    }

    private boolean checkMicroFull(int[] board, int mx, int my) {
        int sx = mx * 3, sy = my * 3;
        for (int lx = 0; lx < 3; lx++)
            for (int ly = 0; ly < 3; ly++)
                if (board[(sx+lx)*9+sy+ly] == 0) return false;
        return true;
    }

    private boolean checkMacroFull(int[] macro) {
        for (int i = 0; i < 9; i++)
            if (macro[i] == 0 || macro[i] == 4) return false;
        return true;
    }

    //  Zobrist hashing  (full re-computation – fast enough for this branching)

    private long computeHash(int[] board, int[] macro, int sideToMove) {
        long h = 0;
        for (int i = 0; i < 81; i++)
            if (board[i] > 0) h ^= ZOB_BOARD[i][board[i] - 1];
        for (int i = 0; i < 9; i++) {
            int s = macro[i];
            if (s > 0) h ^= ZOB_MACRO[i][s - 1];   // states 1-4 → indices 0-3
        }
        if (sideToMove == 1) h ^= ZOB_SIDE;
        return h;
    }

    //  Transposition Table  (two-tier: depth-preferred + always-replace)

    private void ttStore(long hash, int depth, int score, int type, IMove move) {
        int base = (int)(hash & TT_MASK) * 2;  // two slots per bucket

        // Slot 0: depth-preferred
        if (TT_KEY[base] == 0 || depth >= (TT_DEPTH[base] & 0xFF)) {
            TT_KEY  [base] = hash;
            TT_SCORE[base] = score;
            TT_DEPTH[base] = (byte) Math.min(depth, 127);
            TT_TYPE [base] = (byte) type;
            TT_MOVEX[base] = (byte)(move != null ? move.getX() : -1);
            TT_MOVEY[base] = (byte)(move != null ? move.getY() : -1);
        }
        // Slot 1: always-replace
        int slot1 = base + 1;
        TT_KEY  [slot1] = hash;
        TT_SCORE[slot1] = score;
        TT_DEPTH[slot1] = (byte) Math.min(depth, 127);
        TT_TYPE [slot1] = (byte) type;
        TT_MOVEX[slot1] = (byte)(move != null ? move.getX() : -1);
        TT_MOVEY[slot1] = (byte)(move != null ? move.getY() : -1);
    }

    /** Returns [score] if the TT entry is usable, null otherwise. */
    private int[] ttProbe(long hash, int depth, int alpha, int beta) {
        int base = (int)(hash & TT_MASK) * 2;
        for (int slot = base; slot <= base + 1; slot++) {
            if (TT_KEY[slot] != hash) continue;
            if ((TT_DEPTH[slot] & 0xFF) < depth) continue;
            int score = TT_SCORE[slot];
            int type  = TT_TYPE [slot];
            if (type == TT_EXACT) return new int[]{score};
            if (type == TT_LOWER && score >= beta) return new int[]{score};
            if (type == TT_UPPER && score <= alpha) return new int[]{score};
        }
        return null;
    }

    private int ttProbeScore(long hash, int depth) {
        int base = (int)(hash & TT_MASK) * 2;
        for (int slot = base; slot <= base + 1; slot++) {
            if (TT_KEY[slot] == hash && (TT_DEPTH[slot] & 0xFF) >= depth && TT_TYPE[slot] == TT_EXACT)
                return TT_SCORE[slot];
        }
        return 0;
    }

    /** Returns the best move stored in TT for this hash, or null. */
    private IMove ttProbeMove(long hash) {
        int base = (int)(hash & TT_MASK) * 2;
        for (int slot = base; slot <= base + 1; slot++) {
            if (TT_KEY[slot] == hash) {
                int bx = TT_MOVEX[slot];
                int by = TT_MOVEY[slot];
                if (bx >= 0) return new Move(bx, by);
            }
        }
        return null;
    }

    //  Move ordering

    /**
     * Score moves for ordering at internal nodes.
     * Priority: TT move > win > block > killer 0 > killer 1 > history > positional
     */
    private void orderMoves(List<IMove> moves, int[] board, int[] macro, IMove ttMove, int ply) {
        int currentPlayer = (myPlayer + ply) % 2;
        int pval  = currentPlayer + 1;
        int opval = (currentPlayer ^ 1) + 1;

        moves.sort((a, b) -> {
            int sa = scoreMoveInternal(a, board, macro, pval, opval, ttMove, ply);
            int sb = scoreMoveInternal(b, board, macro, pval, opval, ttMove, ply);
            return sb - sa;
        });
    }

    /** Same but at root where we also have TT info. */
    private int[] orderMovesRoot(List<IMove> moves, int[] board, int[] macro, long hash) {
        IMove ttMove = ttProbeMove(hash);
        int pval  = myPlayer + 1;
        int opval = (myPlayer ^ 1) + 1;

        int[] scores = new int[moves.size()];
        for (int i = 0; i < moves.size(); i++) {
            scores[i] = scoreMoveInternal(moves.get(i), board, macro, pval, opval, ttMove, 0);
        }

        // Sort by scores descending using index sort
        Integer[] idx = new Integer[moves.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> scores[b] - scores[a]);

        List<IMove> sorted = new java.util.ArrayList<>(moves.size());
        int[] sortedScores = new int[moves.size()];
        for (int i = 0; i < idx.length; i++) {
            sorted.add(moves.get(idx[i]));
            sortedScores[i] = scores[idx[i]];
        }
        moves.clear();
        moves.addAll(sorted);
        return sortedScores;
    }

    private int scoreMoveInternal(IMove move, int[] board, int[] macro, int pval, int opval, IMove ttMove, int ply) {
        int score = 0;
        int x = move.getX(), y = move.getY();
        int mx = x / 3, my = y / 3;
        int lx = x % 3, ly = y % 3;

        // TT / PV move
        if (ttMove != null && ttMove.getX() == x && ttMove.getY() == y) score += 2_000_000;

        // Winning this micro-board
        if (wouldWinMicro(board, x, y, pval)) score += 1_000_000;

        // Blocking opponent micro-win
        if (wouldWinMicro(board, x, y, opval)) score += 500_000;

        // Killers
        if (ply < MAX_DEPTH) {
            int encoded = x * 9 + y;
            if (killers[ply][0] == encoded) score += 90_000;
            else if (killers[ply][1] == encoded) score += 80_000;
        }

        // History heuristic
        int fromCell = mx * 3 + my;
        int toCell   = lx * 3 + ly;
        score += Math.min(history[fromCell][toCell], 70_000);

        // Positional: local cell position
        if (lx == 1 && ly == 1) score += 300; // local center
        else if ((lx & 1) == 0 && (ly & 1) == 0) score += 150; // local corner
        else score += 50; // local edge

        // Macro position of the board we play in
        if (mx == 1 && my == 1) score += 200;
        else if ((mx & 1) == 0 && (my & 1) == 0) score += 100;

        // Penalise sending opponent to the macro center if it's still open
        int tx = x % 3, ty = y % 3;
        int targetMacro = tx * 3 + ty;
        if (macro[targetMacro] == 0 || macro[targetMacro] == 4) {
            if (tx == 1 && ty == 1) score -= 150; // sending them to center
        } else {
            score += 200; // send to already-won board = free move for us next
        }
        return score;
    }

    /** Quick check: would placing pval at (x,y) win the micro-board? */
    private boolean wouldWinMicro(int[] board, int x, int y, int pval) {
        int[] nb = board.clone();
        nb[x * 9 + y] = pval;
        return checkMicroWin(nb, x / 3, y / 3, pval);
    }

    //  Killer & History updates

    private void updateKillers(IMove move, int ply) {
        if (ply >= MAX_DEPTH) return;
        int encoded = move.getX() * 9 + move.getY();
        if (killers[ply][0] != encoded) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = encoded;
        }
    }

    private void updateHistory(IMove move, int depth) {
        int mx = move.getX() / 3, my = move.getY() / 3;
        int lx = move.getX() % 3, ly = move.getY() % 3;
        history[mx * 3 + my][lx * 3 + ly] += depth * depth;
    }

    //  Static evaluation  (from the current side-to-move's perspective)

    /**
     * Returns a score from the perspective of the player who just moved
     * (positive = good for that player, negative = bad).
     * The "current player" about to move is (myPlayer + ply) % 2,
     * but at leaf nodes we evaluate from myPlayer's absolute perspective
     * and negate in the negamax tree as needed.
     * We evaluate from myPlayer's perspective always; negamax handles negation.
     */
    private int staticEval(int[] board, int[] macro) {
        int me  = myPlayer + 1;    // 1 or 2
        int opp = (myPlayer ^ 1) + 1;

        int score = 0;

        // Macro board evaluation
        score += evalMacroGrid(macro, me, opp);

        // Micro board evaluation
        for (int mx = 0; mx < 3; mx++) {
            for (int my = 0; my < 3; my++) {
                int mState = macro[mx * 3 + my];
                // Only evaluate contested (active or open) micro-boards
                if (mState == 1 || mState == 2 || mState == 3) continue;

                int macroWeight = macroWeight(mx, my);
                score += evalMicroGrid(board, mx, my, me, opp) * macroWeight;
            }
        }

        return score;
    }

    /**
     * Score the 3×3 macro grid for player me vs opp.
     * Uses the integer macro array where 1 = me-wins, 2 = opp-wins, 3 = tie, 4 = active, 0 = open.
     */
    private int evalMacroGrid(int[] macro, int me, int opp) {
        int score = 0;

        // Open-line scoring on the macro board
        score += scoreLines3x3(macro, me, opp, 0,
                /*two*/ 800, /*one*/ 200, /*center*/ 150, /*corner*/ 60);

        // Already-won macro cells contribute positional bonuses
        for (int i = 0; i < 9; i++) {
            int mx = i / 3, my = i % 3;
            if (macro[i] == me)  score += 400 + macroWeight(mx, my) * 50;
            if (macro[i] == opp) score -= 400 + macroWeight(mx, my) * 50;
        }

        return score;
    }

    /**
     * Score a micro-board at (mx,my) within the 9×9 board.
     */
    private int evalMicroGrid(int[] board, int mx, int my, int me, int opp) {
        int sx = mx * 3, sy = my * 3;
        // Build a temp 9-element array representing this 3x3 grid
        int[] local = new int[9];
        for (int lx = 0; lx < 3; lx++)
            for (int ly = 0; ly < 3; ly++)
                local[lx * 3 + ly] = board[(sx + lx) * 9 + sy + ly];

        return scoreLines3x3(local, me, opp, 0,
                /*two*/ 25, /*one*/ 7, /*center*/ 10, /*corner*/ 4);
    }

    /**
     * Generic open-line scorer for a 9-element 3×3 grid encoded as grid[row*3+col].
     * 'skip' = value to ignore (0 = empty, 4 = active are both considered open).
     */
    private int scoreLines3x3(int[] g, int me, int opp, int skip,
                              int twoScore, int oneScore,
                              int centerBonus, int cornerBonus) {
        int score = 0;
        // All 8 lines: 3 rows, 3 cols, 2 diags
        int[][] lines = {
                {0,1,2},{3,4,5},{6,7,8}, // rows
                {0,3,6},{1,4,7},{2,5,8}, // cols
                {0,4,8},{2,4,6} // diags
        };
        for (int[] line : lines) {
            int meC = 0, opC = 0;
            for (int idx : line) {
                int v = g[idx];
                if (v == me)  meC++;
                else if (v == opp) opC++;
                    // v == 0 (open), v==4 (active/available) → neither player owns it, leave counts as-is
                    // v == 3 (TIE) → blocks both players on this line
                else if (v == 3) { meC = 3; opC = 3; break; }
            }
            if (opC == 0) {
                if (meC == 2) score += twoScore;
                else if (meC == 1) score += oneScore;
            }
            if (meC == 0) {
                if (opC == 2) score -= twoScore;
                else if (opC == 1) score -= oneScore;
            }
        }
        // Center
        if (g[4] == me)  score += centerBonus;
        if (g[4] == opp) score -= centerBonus;
        // Corners
        for (int c : new int[]{0, 2, 6, 8}) {
            if (g[c] == me)  score += cornerBonus;
            if (g[c] == opp) score -= cornerBonus;
        }
        return score;
    }

    /** Weight for a macro cell position (center > corner > edge). */
    private int macroWeight(int mx, int my) {
        if (mx == 1 && my == 1) return 4;
        if ((mx & 1) == 0 && (my & 1) == 0) return 3;
        return 2;
    }

    private boolean isDecisiveScore(int score) {
        return Math.abs(score) >= WIN / 2;
    }

    //  Utility

    private boolean timeout() {
        return (System.currentTimeMillis() - startTime) >= timeLimitMs;
    }

    @Override
    public String getBotName() { return BOT_NAME; }
}