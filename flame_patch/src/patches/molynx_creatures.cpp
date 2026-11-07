//
// Molynx mod (https://github.com/FrimpFrog/molynx-dk2-mod), GPL-3.0.
//
// Diagnostics while the game learns a 47th creature type:
//  - flame\molynx.log: what this patch did, with times;
//  - a watchdog thread: when the game window stops responding for 12 s (the game hangs), every
//    thread's stack is written to flame\StackInfo\Flame-StackInfo-*.txt (three samples, 2 s
//    apart), so the place it hangs in can be found;
//  - Ctrl+Shift+F12 writes the same stack file at any time.
//

#include <Windows.h>
#include <cstdarg>
#include <cstdio>
#include <thread>
#include "patches/molynx_creatures.h"
#include "tools/bug_hunter.h"

namespace {
    DWORD g_pid = 0;
    DWORD g_startTick = 0;

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
    log("Flame with the Molynx patches: started");
    std::thread(watchdog).detach();
}
