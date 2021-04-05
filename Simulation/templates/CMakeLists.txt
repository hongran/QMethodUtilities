# CMakeLists.txt file for runtime files

# WARNING - ADDING A FILE TO THIS DIRECTORY WILL NOT TRIGGER
# A REBUILD (since it doesn't change the CMakeLists.txt file).

# install all *.fcl files in this directory
file(GLOB rt_files *.json *.g2map *.dat *.root)
install( FILES ${rt_files}
    DESTINATION ${product}/${version}/runTimeFiles/templatesRun1/beamTemplates/calo10BeamTemplates )

# Also install in the build area
foreach(aFile ${rt_files})
  get_filename_component( basename ${aFile} NAME )
  configure_file( ${aFile}
      ${CMAKE_BINARY_DIR}/${product}/runTimeFiles/templatesRun1/beamTemplates/calo10BeamTemplates/${basename} COPYONLY )
endforeach(aFile)
