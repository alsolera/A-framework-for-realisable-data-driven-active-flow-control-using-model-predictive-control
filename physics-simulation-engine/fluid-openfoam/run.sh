#!/bin/bash
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
#------------------------------------------------------------------------------
## parallel run
mpirun -np 64 --bind-to none --allow-run-as-root pimpleFoam  -parallel > log.pimpleFoam 2>&1 &

## single run
#runApplication $(getApplication)
