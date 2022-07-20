file(REMOVE_RECURSE
  "libfoo.a"
  "libfoo.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/foo.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
