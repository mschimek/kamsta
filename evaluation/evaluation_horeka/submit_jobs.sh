#!/bin/bash

jobfile_dir=$1

cd ${jobfile_dir}

for file in *
do
  sbatch $file
  if [ $? -eq 0 ]
  then
    echo "successfully submitted ${fie}"
    mv $file ../submitted_jobs/$file
  fi
done

cd -

