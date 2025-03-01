#!/bin/bash
#SBATCH --job-name=tvm-metaschedule         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=v100        
#SBATCH --time=00:50:00          # total run time limit (HH:MM:SS)
#disableSBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=nk2747@princeton.edu


module purge
source ~/.bashrc
echo $(module list)
conda activate tvm-build-venv
python -c "import tvm; print(tvm.__file__)"
python -c "import tvm; print(tvm.cuda().exist)"
CUDA_LAUNCH_BLOCKING=1 python tune_te.py --workload C2D --num-trials 200 --number 20 --work-dir /home/nk2747/tvm/python/tvm/meta_schedule/testing/workdir/newest --cpu-flush False

