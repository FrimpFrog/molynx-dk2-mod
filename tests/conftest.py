# tests/blender/ runs inside Blender (needs bpy), so plain pytest must not collect it.
collect_ignore = ["blender"]
