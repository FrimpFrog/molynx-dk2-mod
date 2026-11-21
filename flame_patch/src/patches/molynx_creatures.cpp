//
// Molynx mod (https://github.com/FrimpFrog/molynx-dk2-mod), GPL-3.0.
//
// Creature types beyond the game's 46 (Molynx is 47; 48, 49, 50... the same way).
//
// What the game (DKII.EXE 1.70) does with creature types, and what this patch changes. The game
// loads and looks up creature types by count (CWorld::fun_50D390 searches the sorted table, the
// panels allocate per count), so a 47th type loads and plays. Two things are sized for 46:
//
// 1. Each player's portal list, CPlayer::fB57_creatureEntranceInfo[46] (18 bytes each: the
//    creature type at +0, timers at +2, +6, +0xA, +0xE). CPlayer::init (0x4B8640) fills slot i
//    with the type of creature-table index i (CWorld vtable +0xB8) for i < 46, so the creature at
//    index 46 and up (type 47 and up) is never in the list and never comes through a portal. The
//    portal code (sub_4DBD90 and Entrance_cpp_4DBF90) walks slots 0..45 and again asks +0xB8 for
//    each slot's type. This patch points those five +0xB8 calls at slotType(): with 46 creature
//    types or fewer it is the game's own answer (nothing changes); with more, the slots drop the
//    highest-indexed types that can never come through any entrance (not AVAILABLE_VIA_PORTAL, or
//    IS_UNIQUE: heroes, Jack, the unique creatures) and take the new types in their place, so
//    every type a portal can bring has a slot (46 slots, 13 portal types in the game).
// 2. The "seen this type" flag (sub_4C0480 reads, CPlayer::fun_4C04A0 sets): it is stored in the
//    portal list at player + 0xB58 + 18 * type, which for type 47 and up is past the list's end
//    (type 47 writes into fE93_possessedCreatureInitialGold, 48..50 into fE97_roomRemovalInfos).
//    Both are replaced: types below 47 use the game's own byte, higher types a table here.
//
// Diagnostics: flame\molynx.log (the slot table when it changes); a watchdog that writes every
// thread's stack to flame\StackInfo when the game window stops responding for 12 s; Ctrl+Shift+F12
// writes the same at any time.
//

#include <Windows.h>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "patches/molynx_creatures.h"
#include "tools/bug_hunter.h"

namespace {
    DWORD g_pid = 0;
    DWORD g_startTick = 0;

    // ------------------------------------------------------------------ the game's functions
    constexpr int GAME_SLOTS = 46;                  // CPlayer::fB57_creatureEntranceInfo[46]
    constexpr int FIRST_EXTRA_TYPE = 47;            // types whose "seen" flag is past the list
    constexpr uint32_t FLAG_AVAILABLE_VIA_PORTAL = 0x400, FLAG_IS_UNIQUE = 0x8000000;
    constexpr uint32_t DATA_FLAGS_AT = 0x686;       // MyCreatureDataObj flags (loadCreatureType)

    typedef uint32_t (__thiscall *count_fn)(void *world);                   // CWorld vtable +0xC4
    typedef uint32_t (__thiscall *type_by_index_fn)(void *world, uint32_t);  // CWorld vtable +0xB8
    typedef uint8_t *(__thiscall *data_by_type_fn)(void *world, uint32_t);   // CWorld vtable +0xC0
    typedef uint32_t (__thiscall *game_tick_fn)(void *world);                // getGameTick 0x517490

    void **vtable(void *obj) { return *(void ***) obj; }
    uint32_t creatureCount(void *w) { return ((count_fn) vtable(w)[0xC4 / 4])(w); }
    uint8_t typeByIndex(void *w, uint32_t i) { return (uint8_t) ((type_by_index_fn) vtable(w)[0xB8 / 4])(w, i); }
    uint8_t *dataByType(void *w, uint32_t t) { return ((data_by_type_fn) vtable(w)[0xC0 / 4])(w, t); }

    void **const g_pCWorld = (void **) 0x006E5054;
    uint32_t gameTick() {
        void *w = *g_pCWorld;
        return w ? ((game_tick_fn) 0x00517490)(w) : 0;
    }

    // ------------------------------------------------------------------ 1. the portal slots
    std::vector<uint32_t> g_slots;      // slot -> creature-table index (when there are > 46 types)
    std::string g_lastLogged;

    bool neverByEntrance(void *w, uint8_t type) {
        uint8_t *data = dataByType(w, type);
        if (!data) return true;
        uint32_t flags = *(uint32_t *) (data + DATA_FLAGS_AT);
        return !(flags & FLAG_AVAILABLE_VIA_PORTAL) || (flags & FLAG_IS_UNIQUE);
    }

    void buildSlots(void *w) {
        uint32_t count = creatureCount(w);
        g_slots.clear();
        if (count <= GAME_SLOTS) return;
        uint32_t extra = count - GAME_SLOTS;
        std::vector<bool> dropped(GAME_SLOTS, false);
        for (int i = GAME_SLOTS - 1; i >= 0 && extra > 0; --i) {   // highest indexes first
            if (neverByEntrance(w, typeByIndex(w, i))) { dropped[i] = true; --extra; }
        }
        for (uint32_t i = 0; i < GAME_SLOTS; ++i) if (!dropped[i]) g_slots.push_back(i);
        for (uint32_t i = GAME_SLOTS; i < count && g_slots.size() < GAME_SLOTS; ++i) g_slots.push_back(i);
        std::string line = "portal slots (" + std::to_string(count) + " creature types):";
        for (uint32_t i : g_slots) line += " " + std::to_string(typeByIndex(w, i));
        if (line != g_lastLogged) {
            g_lastLogged = line;
            patch::molynx_creatures::log("%s", line.c_str());
        }
    }

    // Called in place of `call [vtable + 0xB8]` (thiscall world, slot; ret 4); returns the type in al.
    uint32_t __fastcall slotType(void *world, void *, uint32_t slot) {
        if (slot == 0) buildSlots(world);  // every caller walks the slots from 0 upwards
        if (g_slots.empty()) return typeByIndex(world, slot);
        if (slot >= g_slots.size()) return 0;
        return typeByIndex(world, g_slots[slot]);
    }

    // ------------------------------------------------------------------ 2. the "seen" flags
    struct Seen { std::array<uint32_t, 256> setAtTick{}; };  // 0 = not set; else game tick + 1
    std::unordered_map<uint8_t *, Seen> g_seen;

    uint32_t __fastcall getSeen(uint8_t *player, void *, uint32_t type) {   // sub_4C0480
        type &= 0xFF;
        if (type < FIRST_EXTRA_TYPE) return player[0xB58 + 18 * type];
        uint32_t at = g_seen[player].setAtTick[type];
        return (at != 0 && at - 1 <= gameTick()) ? 1 : 0;  // set later than now: a previous level
    }

    uint32_t __fastcall setSeen(uint8_t *player, void *, uint32_t type) {   // CPlayer::fun_4C04A0
        type &= 0xFF;
        if (type < FIRST_EXTRA_TYPE) {
            player[0xB58 + 18 * type] = 1;
            return 9 * type;
        }
        g_seen[player].setAtTick[type] = gameTick() + 1;
        return 9 * type;
    }

    // ------------------------------------------------------------------ patching
    bool write(uintptr_t at, const std::vector<uint8_t> &expect, const std::vector<uint8_t> &bytes,
               const char *what) {
        if (memcmp((void *) at, expect.data(), expect.size()) != 0) {
            patch::molynx_creatures::log("NOT patched %s at %08X: the game's bytes differ", what, (unsigned) at);
            return false;
        }
        DWORD old;
        VirtualProtect((void *) at, bytes.size(), PAGE_EXECUTE_READWRITE, &old);
        memcpy((void *) at, bytes.data(), bytes.size());
        VirtualProtect((void *) at, bytes.size(), old, &old);
        FlushInstructionCache(GetCurrentProcess(), (void *) at, bytes.size());
        return true;
    }

    std::vector<uint8_t> rel32(uint8_t op, uintptr_t at, void *to, size_t len) {
        std::vector<uint8_t> b(len, 0x90);
        b[0] = op;
        int32_t rel = (int32_t) ((uintptr_t) to - (at + 5));
        memcpy(&b[1], &rel, 4);
        return b;
    }

    void patchGame() {
        // the five +0xB8 calls on the portal slots: `call dword ptr [reg + 0xB8]` (6 bytes)
        struct Site { uintptr_t at; uint8_t modrm; const char *what; };
        const Site sites[] = {
            {0x004B8BB5, 0x90, "CPlayer::init slot fill"},
            {0x004DBDD0, 0x92, "sub_4DBD90 slot loop"},
            {0x004DC00D, 0x90, "Entrance_cpp_4DBF90 candidates"},
            {0x004DC5FE, 0x90, "Entrance_cpp_4DBF90 timer loop 1"},
            {0x004DC6A3, 0x90, "Entrance_cpp_4DBF90 timer loop 2"},
        };
        int ok = 0;
        for (auto &s : sites) {
            ok += write(s.at, {0xFF, s.modrm, 0xB8, 0x00, 0x00, 0x00}, rel32(0xE8, s.at, (void *) slotType, 6), s.what);
        }
        // the "seen" flag functions: their first instruction (mov eax, [esp+4]) becomes a jmp
        ok += write(0x004C0480, {0x8B, 0x44, 0x24, 0x04, 0x33, 0xD2}, rel32(0xE9, 0x004C0480, (void *) getSeen, 5), "sub_4C0480 (get seen)");
        ok += write(0x004C04A0, {0x8B, 0x44, 0x24, 0x04, 0x25}, rel32(0xE9, 0x004C04A0, (void *) setSeen, 5), "fun_4C04A0 (set seen)");
        patch::molynx_creatures::log("creature types beyond 46: %d of 7 patches applied", ok);
    }

    // ------------------------------------------------------------------ diagnostics
    BOOL CALLBACK findGameWindow(HWND hwnd, LPARAM out) {
        DWORD pid = 0;
        GetWindowThreadProcessId(hwnd, &pid);
        if (pid != g_pid || !IsWindowVisible(hwnd) || GetWindow(hwnd, GW_OWNER) != NULL) return TRUE;
        *(HWND *) out = hwnd;
        return FALSE;
    }

    HWND gameWindow() {
        HWND hwnd = NULL;
        EnumWindows(findGameWindow, (LPARAM) &hwnd);
        return hwnd;
    }

    void dumpStacks(const char *why, int samples) {
        for (int i = 0; i < samples; ++i) {
            patch::molynx_creatures::log("stack dump %d/%d (%s)", i + 1, samples, why);
            bug_hunter::collectStackInfo();
            if (i + 1 < samples) Sleep(2000);
        }
    }

    void watchdog() {
        int hungSeconds = 0;
        bool dumped = false;
        bool keyWasDown = false;
        while (true) {
            Sleep(1000);
            bool keyDown = (GetAsyncKeyState(VK_F12) & 0x8000) && (GetAsyncKeyState(VK_CONTROL) & 0x8000)
                           && (GetAsyncKeyState(VK_SHIFT) & 0x8000);
            if (keyDown && !keyWasDown) dumpStacks("Ctrl+Shift+F12", 1);
            keyWasDown = keyDown;
            HWND hwnd = gameWindow();
            if (hwnd != NULL && IsHungAppWindow(hwnd)) {
                ++hungSeconds;
                if (hungSeconds >= 12 && !dumped) {
                    dumped = true;
                    dumpStacks("the game window has not responded for 12 s", 3);
                }
            } else {
                if (dumped) patch::molynx_creatures::log("the game window responds again");
                hungSeconds = 0;
                dumped = false;
            }
        }
    }
}

void patch::molynx_creatures::log(const char *format, ...) {
    char path[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, path);
    strcat(path, "\\flame\\molynx.log");
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "%8.3f ", (GetTickCount() - g_startTick) / 1000.0);
    va_list args;
    va_start(args, format);
    vfprintf(f, format, args);
    va_end(args);
    fputc('\n', f);
    fclose(f);
}

void patch::molynx_creatures::init() {
    g_pid = GetCurrentProcessId();
    g_startTick = GetTickCount();
    log("Flame with the Molynx patches v2: started");
    patchGame();
    std::thread(watchdog).detach();
}
