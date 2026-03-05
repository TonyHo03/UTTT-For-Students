package dk.easv.bll.bot;

import dk.easv.bll.field.IField;
import dk.easv.bll.game.IGameState;
import dk.easv.bll.move.IMove;
import dk.easv.bll.move.Move;

import java.util.*;

/**
 * Negamax + Iterative Deepening + Alpha-Beta + Transposition Table
 */
public class SupaNegaBot implements IBot {

    private static final String BOT_NAME = "SupaNegaBot";
    private static final long TIME_MS = 940; // leave 60ms margin
    private static final int MAX_DEPTH = 14; // iterative deepening cap
    private static final int INF = 1_000_000; // represent infinity for our search algorithm
    private static final int WIN_SCORE = 900_000; // terminal win value
    private long deadline; // time management

    // Best-move cache: maps Zobrist hash → best move coords packed as (x<<8|y)
    // Populated during search, consulted at root and in move ordering for killer-move-style hints
    private static final int BM_SIZE = 1 << 20; // ~1M entries
    private final long[] bmKey   = new long[BM_SIZE];
    private final short[] bmMove = new short[BM_SIZE]; // packed (x<<4)|y

    // Transposition table
    private static final int TT_SIZE = 1 << 22; // ~4M entries
    private final long[] ttKey = new long[TT_SIZE];
    private final int[] ttScore = new int[TT_SIZE];
    private final byte[] ttDepth = new byte[TT_SIZE];
    private final byte[] ttFlag = new byte[TT_SIZE]; // 0=exact,1=lower,2=upper

    // Zobrist hashing
    private static final long[][][] ZOBRIST = new long[9][9][3]; // [x][y][player+1]
    private static final long ZOBRIST_TURN;
    static {
        Random r = new Random(0xDEADBEEF_CAFEBABEL);
        for (int x = 0; x < 9; x++)
            for (int y = 0; y < 9; y++)
                for (int p = 0; p < 3; p++)
                    ZOBRIST[x][y][p] = r.nextLong();
        ZOBRIST_TURN = r.nextLong();
    }

    /**
     * Gets legal moves
     * Plays center if possible (opening book)
     * Sets a time deadline (so it doesnt go over 1 sec (1000 ms))
     * Builds our internal GameState
     * Runs iterative deepening Negamax
     * Returns the best move found before time runs out
     */
    @Override
    public IMove doMove(IGameState state) {
        List<IMove> avail = state.getField().getAvailableMoves();

        // Safety guard: if no moves available return null
        if (avail == null || avail.isEmpty()) return null;

        // Opening book: play center of center board if available
        for (IMove m : avail)
            if (m.getX() == 4 && m.getY() == 4) return m;

        deadline = System.currentTimeMillis() + TIME_MS;

        // Build internal game state
        GameState gs = GameState.fromInterface(state);

        // Always have a valid fallback -- first available move
        IMove lastBest = avail.get(0);

        // Seed move ordering with cached best move from a previous search on this position
        IMove cachedBest = lookupBestMove(gs.hash);
        if (cachedBest != null) lastBest = cachedBest;

        // Iterative deepening
        for (int depth = 1; depth <= MAX_DEPTH; depth++) {
            if (System.currentTimeMillis() >= deadline) break;

            SearchResult result = negamax(gs, depth, -INF, INF, true);

            if (result.timedOut) break; // discard incomplete search, lastBest stays valid

            if (result.move != null) {
                lastBest = result.move;
                storeBestMove(gs.hash, result.move); // persist for future turns / rematches
            }

            // Found a forced win -- no need to search deeper
            if (result.score >= WIN_SCORE / 2) break;
        }

        return lastBest;
    }

    @Override
    public String getBotName() { return BOT_NAME; }

    // ── Best-move cache ────────────────────────────────────────────────────────
    /** Store the best move for a given position hash. */
    private void storeBestMove(long hash, IMove move) {
        int idx = (int)(hash & (BM_SIZE - 1));
        bmKey[idx]  = hash;
        bmMove[idx] = (short)((move.getX() << 4) | move.getY());
    }

    /** Look up a previously stored best move for this hash; returns null on miss. */
    private IMove lookupBestMove(long hash) {
        int idx = (int)(hash & (BM_SIZE - 1));
        if (bmKey[idx] != hash) return null;
        int packed = bmMove[idx] & 0xFFFF;
        return new Move(packed >> 4, packed & 0xF);
    }

    // Negamax Algorithm
    /**
     * Class for holding score, best move and timeout flag
     */
    private static class SearchResult {
        int score;
        IMove move;
        boolean timedOut;
        SearchResult(int score, IMove move, boolean timedOut) {
            this.score = score; this.move = move; this.timedOut = timedOut;
        }
    }

    private SearchResult negamax(GameState gs, int depth, int alpha, int beta, boolean isRoot) {
        if (System.currentTimeMillis() >= deadline)
            return new SearchResult(0, null, true);

        // Terminal check
        int terminal = gs.terminalValue();
        if (terminal != Integer.MIN_VALUE) {
            return new SearchResult(terminal, null, false);
        }

        // Depth limit → evaluate
        if (depth == 0) {
            return new SearchResult(evaluate(gs), null, false);
        }

        // Transposition table lookup
        long hash = gs.hash;
        int ttIdx = (int)(hash & (TT_SIZE - 1));
        int ttAlphaOrig = alpha;
        if (ttKey[ttIdx] == hash && ttDepth[ttIdx] >= depth) {
            int s = ttScore[ttIdx];
            byte flag = ttFlag[ttIdx];
            if (flag == 0) { // exact
                if (isRoot) {
                    // Can't use TT move as root move easily here without storing it; just use score to guide
                } else return new SearchResult(s, null, false);
            } else if (flag == 1 && s > alpha) { // lower bound
                alpha = s;
            } else if (flag == 2 && s < beta) { // upper bound
                beta = s;
            }
            if (alpha >= beta && !isRoot)
                return new SearchResult(s, null, false);
        }

        // Generate and order moves
        List<IMove> moves = gs.getAvailableMoves();
        orderMoves(gs, moves);

        int bestScore = -INF;
        IMove bestMove = moves.isEmpty() ? null : moves.get(0);

        for (IMove move : moves) {
            if (System.currentTimeMillis() >= deadline)
                return new SearchResult(bestScore, bestMove, true);

            gs.applyMove(move);
            SearchResult child = negamax(gs, depth - 1, -beta, -alpha, false);
            gs.undoMove(move);

            if (child.timedOut) return new SearchResult(bestScore, bestMove, true);

            int score = -child.score;
            if (score > bestScore) {
                bestScore = score;
                bestMove  = move;
            }
            if (score > alpha) alpha = score;
            if (alpha >= beta) break; // beta cutoff
        }

        // Store in TT
        byte flag;
        if (bestScore <= ttAlphaOrig) flag = 2;  // upper bound
        else if (bestScore >= beta) flag = 1;    // lower bound
        else flag = 0;                           // exact
        ttKey[ttIdx] = hash;
        ttScore[ttIdx] = bestScore;
        ttDepth[ttIdx] = (byte) Math.min(depth, 127);
        ttFlag[ttIdx] = flag;

        // Cache best move for future move ordering / root reuse
        if (bestMove != null) storeBestMove(hash, bestMove);

        return new SearchResult(bestScore, bestMove, false);
    }

    // Move Ordering
    /**
     * Order moves best-first for alpha-beta efficiency:
     *  1. Cached best move from previous search (transposition hint)
     *  2. Macro-winning moves (immediate game win)
     *  3. Opponent macro-win blocks
     *  4. Micro-board winning moves
     *  5. Opponent micro-board blocks
     *  6. Macro fork (creates 2+ independent winning threats on macro board)
     *  7. Block opponent macro fork
     *  8. Macro 2-in-a-row
     *  9. Safe destination (don't send opponent to danger)
     * 10. Positional preference
     */
    private void orderMoves(GameState gs, List<IMove> moves) {
        int me  = gs.currentPlayer;
        int opp = 1 - me;

        // Best-move cache hint for this position
        IMove cachedBest = lookupBestMove(gs.hash);

        int[] scores = new int[moves.size()];
        for (int i = 0; i < moves.size(); i++) {
            IMove m = moves.get(i);
            int s = 0;

            // Cached best move from prior search — float it to the top
            if (cachedBest != null && m.getX() == cachedBest.getX() && m.getY() == cachedBest.getY())
                s += 200000;

            if (gs.moveWinsMacro(m, me))  s += 100000;
            if (gs.moveWinsMacro(m, opp)) s += 50000;
            if (gs.moveWinsMicro(m, me))  s += 10000;
            if (gs.moveWinsMicro(m, opp)) s += 5000;

            // Fork scoring: placed between micro-win blocks and 2-in-a-row
            // A fork creates ≥2 simultaneous winning macro threats → very strong
            int myForks  = gs.countMacroForks(m, me);
            int oppForks = gs.countMacroForks(m, opp);
            if (myForks  >= 2) s += 8000; // actively building a fork
            if (oppForks >= 2) s += 4000; // blocking opponent fork

            // Macro 2-in-a-row
            s += gs.macroTwoInRow(m, me) * 400;
            s -= gs.macroTwoInRow(m, opp) * 600;

            // Destination safety
            int destMx = m.getX() % 3, destMy = m.getY() % 3;
            String dest = gs.macro[destMx][destMy];
            boolean destFree = dest.equals(GameState.EMPTY) || dest.equals(GameState.AVAIL);
            if (destFree) {
                if (gs.boardHasImmediateWin(destMx, destMy, opp)) s -= 3000;
                if (gs.boardHasTwoInRow(destMx, destMy, opp)) s -= 1000;
                if (destMx == 1 && destMy == 1) s -= 200;
            }

            // Positional
            int lx = m.getX() % 3, ly = m.getY() % 3;
            if (lx == 1 && ly == 1) s += 50;
            else if ((lx == 0 || lx == 2) && (ly == 0 || ly == 2)) s += 20;
            int mx = m.getX() / 3, my = m.getY() / 3;
            if (mx == 1 && my == 1) s += 30;

            scores[i] = s;
        }

        // Insertion sort (stable, fast for small lists)
        for (int i = 1; i < moves.size(); i++) {
            IMove keyM = moves.get(i);
            int keyS = scores[i];
            int j = i - 1;
            while (j >= 0 && scores[j] < keyS) {
                moves.set(j + 1, moves.get(j));
                scores[j + 1] = scores[j];
                j--;
            }
            moves.set(j + 1, keyM);
            scores[j + 1] = keyS;
        }
    }

    // Evaluation
    /**
     * Static evaluation from current player's perspective.
     * Positive = good for current player, negative = bad.
     *
     * Weights (tuned to emphasise macro control):
     *   macro win line (2-in-a-row open):    150 each
     *   macro board owned:                    80 each (+40 center, +20 corner)
     *   micro 2-in-a-row in active board:     12 each
     *   center cell of micro board owned:      6 each
     *   destination danger penalty:          -50 per opponent immediate win threat
     */
    private int evaluate(GameState gs) {
        int me  = gs.currentPlayer;
        int opp = 1 - me;
        String myP  = "" + me;
        String oppP = "" + opp;

        int score = 0;

        // ── Macro level ────────────────────────────────────────────
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            String cell = gs.macro[i][j];
            if (cell.equals(myP)) {
                score += 80;
                if (i == 1 && j == 1) score += 40;
                else if ((i == 0 || i == 2) && (j == 0 || j == 2)) score += 20;
            } else if (cell.equals(oppP)) {
                score -= 80;
                if (i == 1 && j == 1) score -= 40;
                else if ((i == 0 || i == 2) && (j == 0 || j == 2)) score -= 20;
            }
        }

        // Macro 2-in-a-row (open lines with 2 mine, 1 empty)
        score += 150 * countMacroLines(gs, myP,  2, true);
        score -= 200 * countMacroLines(gs, oppP, 2, true); // blocking is more valuable

        // Macro fork potential: having ≥2 simultaneous threats is a strong structural advantage
        int myMacroThreats  = gs.macroThreatsPublic(myP);
        int oppMacroThreats = gs.macroThreatsPublic(oppP);
        if (myMacroThreats  >= 2) score += 120 * myMacroThreats;
        if (oppMacroThreats >= 2) score -= 160 * oppMacroThreats; // opponent fork is more dangerous

        // Macro 1-in-a-row (open lines with 1 mine, 2 empty) — softer signal
        score += 20 * countMacroLines(gs, myP,  1, true);
        score -= 25 * countMacroLines(gs, oppP, 1, true);

        // ── Micro level (only in active/undecided macro cells) ──────
        for (int bx = 0; bx < 3; bx++) for (int by = 0; by < 3; by++) {
            String cell = gs.macro[bx][by];
            if (!cell.equals(GameState.EMPTY) && !cell.equals(GameState.AVAIL)) continue;

            int sx = bx * 3, sy = by * 3;

            // Count 2-in-a-rows in this micro board
            score += 12 * microTwoInRows(gs, sx, sy, myP);
            score -= 15 * microTwoInRows(gs, sx, sy, oppP);

            // Center cell bonus
            if (gs.board[sx+1][sy+1].equals(myP))  score += 6;
            if (gs.board[sx+1][sy+1].equals(oppP)) score -= 6;
        }

        // ── Destination danger ──
        // Penalise states where current player just sent opponent somewhere good
        // (Difficult to compute post-hoc; skip in eval — handled by move ordering)

        return score;
    }

    private int countMacroLines(GameState gs, String p, int target, boolean requireOpen) {
        int count = 0;
        String[][] m = gs.macro;
        // Rows
        for (int i = 0; i < 3; i++) {
            int pc = 0, ec = 0, blocked = 0;
            for (int j = 0; j < 3; j++) {
                if (m[i][j].equals(p)) pc++;
                else if (m[i][j].equals(GameState.EMPTY) || m[i][j].equals(GameState.AVAIL)) ec++;
                else blocked++;
            }
            if (pc == target && (!requireOpen || blocked == 0)) count++;
        }
        // Cols
        for (int j = 0; j < 3; j++) {
            int pc = 0, ec = 0, blocked = 0;
            for (int i = 0; i < 3; i++) {
                if (m[i][j].equals(p)) pc++;
                else if (m[i][j].equals(GameState.EMPTY) || m[i][j].equals(GameState.AVAIL)) ec++;
                else blocked++;
            }
            if (pc == target && (!requireOpen || blocked == 0)) count++;
        }
        // Diag
        { int pc=0,ec=0,bl=0; for(int i=0;i<3;i++){if(m[i][i].equals(p))pc++;else if(m[i][i].equals(GameState.EMPTY)||m[i][i].equals(GameState.AVAIL))ec++;else bl++;} if(pc==target&&(!requireOpen||bl==0))count++; }
        // Anti
        { int pc=0,ec=0,bl=0; for(int i=0;i<3;i++){if(m[i][2-i].equals(p))pc++;else if(m[i][2-i].equals(GameState.EMPTY)||m[i][2-i].equals(GameState.AVAIL))ec++;else bl++;} if(pc==target&&(!requireOpen||bl==0))count++; }
        return count;
    }

    private int microTwoInRows(GameState gs, int sx, int sy, String p) {
        int count = 0;
        for (int i = 0; i < 3; i++) {
            int pr=0,er=0,pc=0,ec=0;
            for (int j = 0; j < 3; j++) {
                if (gs.board[sx+i][sy+j].equals(p)) pr++; else if (gs.board[sx+i][sy+j].equals(GameState.EMPTY)) er++;
                if (gs.board[sx+j][sy+i].equals(p)) pc++; else if (gs.board[sx+j][sy+i].equals(GameState.EMPTY)) ec++;
            }
            if (pr==2 && er>=1) count++;
            if (pc==2 && ec>=1) count++;
        }
        { int pd=0,ed=0; for(int i=0;i<3;i++){if(gs.board[sx+i][sy+i].equals(p))pd++;else if(gs.board[sx+i][sy+i].equals(GameState.EMPTY))ed++;} if(pd==2&&ed>=1)count++; }
        { int pd=0,ed=0; for(int i=0;i<3;i++){if(gs.board[sx+i][sy+2-i].equals(p))pd++;else if(gs.board[sx+i][sy+2-i].equals(GameState.EMPTY))ed++;} if(pd==2&&ed>=1)count++; }
        return count;
    }

    // Game State
    /**
     * Fast, undo-capable game state for negamax.
     * Uses a move stack for efficient undo without copying.
     */
    private static class GameState {
        static final String AVAIL = IField.AVAILABLE_FIELD;
        static final String EMPTY = IField.EMPTY_FIELD;

        String[][] board = new String[9][9];
        String[][] macro = new String[3][3];
        int currentPlayer;
        long hash;

        // Undo stack entries: [x, y, prevMacro encoding, prevHash]
        private final Deque<long[]> undoStack = new ArrayDeque<>();
        // Previous macro board state (encoded as 3x3 bytes)
        private final Deque<String[][]> macroUndo = new ArrayDeque<>();

        static GameState fromInterface(IGameState gs) {
            GameState s = new GameState();
            s.currentPlayer = gs.getMoveNumber() % 2;
            for (int i = 0; i < 9; i++) s.board[i] = Arrays.copyOf(gs.getField().getBoard()[i], 9);
            for (int i = 0; i < 3; i++) s.macro[i] = Arrays.copyOf(gs.getField().getMacroboard()[i], 3);
            s.hash = s.computeHash();
            return s;
        }

        private long computeHash() {
            long h = 0;
            for (int x = 0; x < 9; x++)
                for (int y = 0; y < 9; y++) {
                    String c = board[x][y];
                    if (c.equals("0")) h ^= ZOBRIST[x][y][1];
                    else if (c.equals("1")) h ^= ZOBRIST[x][y][2];
                }
            if (currentPlayer == 1) h ^= ZOBRIST_TURN;
            return h;
        }

        List<IMove> getAvailableMoves() {
            List<IMove> moves = new ArrayList<>();
            for (int x = 0; x < 9; x++)
                for (int y = 0; y < 9; y++)
                    if (board[x][y].equals(EMPTY) && macro[x/3][y/3].equals(AVAIL))
                        moves.add(new Move(x, y));
            return moves;
        }

        void applyMove(IMove move) {
            int x = move.getX(), y = move.getY();

            // Save undo info
            String[][] macroCopy = new String[3][3];
            for (int i = 0; i < 3; i++) macroCopy[i] = Arrays.copyOf(macro[i], 3);
            macroUndo.push(macroCopy);
            undoStack.push(new long[]{x, y, hash});

            // Apply
            board[x][y] = "" + currentPlayer;
            hash ^= ZOBRIST[x][y][currentPlayer + 1];

            // Update micro/macro
            int mx = x / 3, my = y / 3;
            if (macro[mx][my].equals(AVAIL) || macro[mx][my].equals(EMPTY)) {
                if (winsLocal(x, y, "" + currentPlayer))   macro[mx][my] = "" + currentPlayer;
                else if (localFull(mx, my))                macro[mx][my] = "TIE";
            }

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    if (macro[i][j].equals(AVAIL)) macro[i][j] = EMPTY;

            int nextMx = x % 3, nextMy = y % 3;
            if (macro[nextMx][nextMy].equals(EMPTY)) {
                macro[nextMx][nextMy] = AVAIL;
            } else {
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        if (macro[i][j].equals(EMPTY)) macro[i][j] = AVAIL;
            }

            hash ^= ZOBRIST_TURN;
            currentPlayer = 1 - currentPlayer;
        }

        void undoMove(IMove move) {
            long[] info = undoStack.pop();
            String[][] prevMacro = macroUndo.pop();
            int x = (int)info[0], y = (int)info[1];
            long prevHash = info[2];

            board[x][y] = EMPTY;
            for (int i = 0; i < 3; i++) macro[i] = Arrays.copyOf(prevMacro[i], 3);
            hash = prevHash;
            currentPlayer = 1 - currentPlayer;
        }

        /**
         * Terminal value from the perspective of the CURRENT player (the one to move next).
         * Negamax convention: positive = good for current player.
         *
         * After applyMove(), currentPlayer has already flipped to the NEXT player.
         * So the player who just moved is (1 - currentPlayer).
         * If that player won → current player LOST → return -WIN_SCORE.
         * Tie → return 0.
         * Not terminal → return Integer.MIN_VALUE sentinel.
         */
        int terminalValue() {
            String justMoved = "" + (1 - currentPlayer);
            if (winsMacroFor(justMoved)) return -WIN_SCORE; // current player lost

            // Check tie: all macro cells resolved, no winner
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    if (macro[i][j].equals(AVAIL) || macro[i][j].equals(EMPTY))
                        return Integer.MIN_VALUE; // still active
            return 0; // tie
        }

        boolean winsMacroFor(String p) {
            for (int i = 0; i < 3; i++) {
                if (macro[i][0].equals(p) && macro[i][1].equals(p) && macro[i][2].equals(p)) return true;
                if (macro[0][i].equals(p) && macro[1][i].equals(p) && macro[2][i].equals(p)) return true;
            }
            return (macro[0][0].equals(p) && macro[1][1].equals(p) && macro[2][2].equals(p))
                    || (macro[0][2].equals(p) && macro[1][1].equals(p) && macro[2][0].equals(p));
        }

        boolean moveWinsMicro(IMove move, int player) {
            int x = move.getX(), y = move.getY();
            if (!board[x][y].equals(EMPTY)) return false;
            board[x][y] = "" + player;
            boolean win = winsLocal(x, y, "" + player);
            board[x][y] = EMPTY;
            return win;
        }

        boolean moveWinsMacro(IMove move, int player) {
            if (!moveWinsMicro(move, player)) return false;
            int mx = move.getX() / 3, my = move.getY() / 3;
            String prev = macro[mx][my];
            macro[mx][my] = "" + player;
            boolean win = winsMacroFor("" + player);
            macro[mx][my] = prev;
            return win;
        }

        int macroTwoInRow(IMove move, int player) {
            if (!moveWinsMicro(move, player)) return 0;
            int mx = move.getX() / 3, my = move.getY() / 3;
            String prev = macro[mx][my];
            macro[mx][my] = "" + player;
            int count = 0;
            String p = "" + player;
            // Count lines with exactly 2 of p and 1 free
            String[][] m = macro;
            for (int i = 0; i < 3; i++) {
                int pr=0,er=0,pc=0,ec=0;
                for (int j = 0; j < 3; j++) {
                    if (m[i][j].equals(p)) pr++; else if (m[i][j].equals(EMPTY)||m[i][j].equals(AVAIL)) er++;
                    if (m[j][i].equals(p)) pc++; else if (m[j][i].equals(EMPTY)||m[j][i].equals(AVAIL)) ec++;
                }
                if (pr==2&&er>=1) count++;
                if (pc==2&&ec>=1) count++;
            }
            { int pd=0,ed=0; for(int i=0;i<3;i++){if(m[i][i].equals(p))pd++;else if(m[i][i].equals(EMPTY)||m[i][i].equals(AVAIL))ed++;} if(pd==2&&ed>=1)count++; }
            { int pd=0,ed=0; for(int i=0;i<3;i++){if(m[i][2-i].equals(p))pd++;else if(m[i][2-i].equals(EMPTY)||m[i][2-i].equals(AVAIL))ed++;} if(pd==2&&ed>=1)count++; }
            macro[mx][my] = prev;
            return count;
        }

        /**
         * Counts how many distinct open macro-winning lines a move creates/extends to ≥2
         * of player's marks — i.e. measures fork potential at the macro level.
         * A "fork" means ≥2 separate lines each having 2-of-player + 1-empty on the macro board.
         * Returns the number of such lines (≥2 means a true fork).
         */
        int countMacroForks(IMove move, int player) {
            if (!moveWinsMicro(move, player)) return 0; // only relevant if this wins a micro-board
            int mx = move.getX() / 3, my = move.getY() / 3;
            String prev = macro[mx][my];
            macro[mx][my] = "" + player;
            int lines = macroThreatsFor("" + player);
            macro[mx][my] = prev;
            return lines;
        }

        /** Count open macro lines with exactly 2-of-p and 1 free (winning threats). */
        private int macroThreatsFor(String p) {
            int count = 0;
            String[][] m = macro;
            for (int i = 0; i < 3; i++) {
                int pr=0,er=0,bl=0,pc=0,ec=0,blc=0;
                for (int j = 0; j < 3; j++) {
                    if (m[i][j].equals(p)) pr++; else if (m[i][j].equals(EMPTY)||m[i][j].equals(AVAIL)) er++; else bl++;
                    if (m[j][i].equals(p)) pc++; else if (m[j][i].equals(EMPTY)||m[j][i].equals(AVAIL)) ec++; else blc++;
                }
                if (pr==2&&er>=1&&bl==0) count++;
                if (pc==2&&ec>=1&&blc==0) count++;
            }
            { int pd=0,ed=0,bl=0; for(int i=0;i<3;i++){if(m[i][i].equals(p))pd++;else if(m[i][i].equals(EMPTY)||m[i][i].equals(AVAIL))ed++;else bl++;} if(pd==2&&ed>=1&&bl==0)count++; }
            { int pd=0,ed=0,bl=0; for(int i=0;i<3;i++){if(m[i][2-i].equals(p))pd++;else if(m[i][2-i].equals(EMPTY)||m[i][2-i].equals(AVAIL))ed++;else bl++;} if(pd==2&&ed>=1&&bl==0)count++; }
            return count;
        }

        /** Public accessor used by evaluate(). */
        int macroThreatsPublic(String p) { return macroThreatsFor(p); }

        boolean boardHasTwoInRow(int bx, int by, int player) {
            String p = "" + player;
            int sx = bx*3, sy = by*3;
            for (int i = 0; i < 3; i++) {
                int pr=0,er=0,pc=0,ec=0;
                for (int j = 0; j < 3; j++) {
                    if(board[sx+i][sy+j].equals(p))pr++;else if(board[sx+i][sy+j].equals(EMPTY))er++;
                    if(board[sx+j][sy+i].equals(p))pc++;else if(board[sx+j][sy+i].equals(EMPTY))ec++;
                }
                if(pr==2&&er>=1)return true;
                if(pc==2&&ec>=1)return true;
            }
            { int pd=0,ed=0; for(int i=0;i<3;i++){if(board[sx+i][sy+i].equals(p))pd++;else if(board[sx+i][sy+i].equals(EMPTY))ed++;} if(pd==2&&ed>=1)return true; }
            { int pd=0,ed=0; for(int i=0;i<3;i++){if(board[sx+i][sy+2-i].equals(p))pd++;else if(board[sx+i][sy+2-i].equals(EMPTY))ed++;} if(pd==2&&ed>=1)return true; }
            return false;
        }

        boolean boardHasImmediateWin(int bx, int by, int player) {
            String p = "" + player;
            int sx = bx*3, sy = by*3;
            for (int x = sx; x < sx+3; x++)
                for (int y = sy; y < sy+3; y++)
                    if (board[x][y].equals(EMPTY)) {
                        board[x][y] = p;
                        boolean win = winsLocal(x, y, p);
                        board[x][y] = EMPTY;
                        if (win) return true;
                    }
            return false;
        }

        private boolean winsLocal(int x, int y, String p) {
            int sx=(x/3)*3, sy=(y/3)*3, lx=x%3, ly=y%3;
            if(board[sx+lx][sy].equals(p)&&board[sx+lx][sy+1].equals(p)&&board[sx+lx][sy+2].equals(p)) return true;
            if(board[sx][sy+ly].equals(p)&&board[sx+1][sy+ly].equals(p)&&board[sx+2][sy+ly].equals(p)) return true;
            if(lx==ly&&board[sx][sy].equals(p)&&board[sx+1][sy+1].equals(p)&&board[sx+2][sy+2].equals(p)) return true;
            if(lx+ly==2&&board[sx][sy+2].equals(p)&&board[sx+1][sy+1].equals(p)&&board[sx+2][sy].equals(p)) return true;
            return false;
        }

        private boolean localFull(int mx, int my) {
            int sx=mx*3, sy=my*3;
            for(int i=sx;i<sx+3;i++) for(int j=sy;j<sy+3;j++) if(board[i][j].equals(EMPTY)) return false;
            return true;
        }
    }
}