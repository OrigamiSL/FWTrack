# Train
python tracking/train.py --script fwtrack --config fwtrack_256_full --save_dir ./output --mode single --nproc_per_node 4 --use_wandb 0

python tracking/train.py --script fwtrack --config fwtrack_256_got --save_dir ./output --mode single --nproc_per_node 4 --use_wandb 0

# Test
# UAV123
python tracking/test.py fwtrack fwtrack_256_full --dataset uav --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1

# LaSOT
python tracking/test.py fwtrack fwtrack_256_full --dataset lasot --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1

# LaSOT_ext
python tracking/test.py fwtrack fwtrack_256_full --dataset lasot_extension_subset --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1

# nfs
python tracking/test.py fwtrack fwtrack_256_full --dataset nfs --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1

# tnl2k
python tracking/test.py fwtrack fwtrack_256_full --dataset tnl2k --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1

# Additional script
# TrackingNet
python tracking/test.py fwtrack fwtrack_256_full --dataset trackingnet --test_checkpoint ./test_checkpoint/FWTrack_best.pth.tar --threads 0 --num_gpus 1
python lib/test/utils/transform_trackingnet.py --tracker_name fwtrack --cfg_name fwtrack_256_full

# Got-10K
python tracking/test.py fwtrack fwtrack_256_got --dataset got10k_test --test_checkpoint 'THE PATH OF YOUR TRAINED CHECKPOINT' --threads 0 --num_gpus 1
python lib/test/utils/transform_got10k.py --tracker_name fwtrack --cfg_name fwtrack_256_got
