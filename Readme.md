This folder currently contains the scene folder.

The code is initialized from [our previous code](https://github.com/darglein/NeAT)
# installation

`conda create -y -n hyperoct python=3.8`

`conda activate hyperoct`

`sh create_env.sh`



# To Optimize

cd build
./bin/hyper_train ../configs/*.ini

replace the * with the file name inside.




# View

We suggest to use vv or imageJ to view the reconstructed volume.

or view using software like vv

change output from hdr to mrc using the python script: 
