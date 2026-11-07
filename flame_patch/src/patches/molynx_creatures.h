//
// Molynx mod (https://github.com/FrimpFrog/molynx-dk2-mod), GPL-3.0.
// Support for creature types beyond the game's 46, and diagnostics for it.
//

#ifndef FLAME_MOLYNX_CREATURES_H
#define FLAME_MOLYNX_CREATURES_H

namespace patch::molynx_creatures {
    void init();
    void log(const char *format, ...);
}

#endif // FLAME_MOLYNX_CREATURES_H
