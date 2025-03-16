import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.analysis.plot_results import print_results_per_video

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8,test_checkpoint = None,update_intervals = None, update_threshold = None,hanning_size = None,\
                pre_seq_number = None,std_weight =None,smooth_type=None,alpha=None,beta=None,double_dayu=None,smooth_thre=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, test_checkpoint,update_intervals,update_threshold,hanning_size,\
                        pre_seq_number,std_weight,smooth_type,alpha,beta,double_dayu,smooth_thre)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)

    print_results_per_video(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),per_video=True)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    parser.add_argument('--test_checkpoint', type=str, default=None)
    parser.add_argument('--update_intervals', type=str, default=None)
    parser.add_argument('--update_threshold', type=str, default=None)
    parser.add_argument('--hanning_size', type=str, default=None)
    parser.add_argument('--pre_seq_number', type=str, default=None)
    parser.add_argument('--std_weight', type=str, default=None)
    parser.add_argument('--smooth_type', type=str, default='single')
    parser.add_argument('--alpha', type=str, default=None)
    parser.add_argument('--beta', type=str, default=None)
    parser.add_argument('--double_dayu', type=str, default=None)
    parser.add_argument('--smooth_thre', type=str, default=None)

    
    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, test_checkpoint = args.test_checkpoint,\
                update_intervals = args.update_intervals, update_threshold = args.update_threshold,hanning_size = args.hanning_size,
                pre_seq_number = args.pre_seq_number,std_weight = args.std_weight,smooth_type=args.smooth_type,\
                alpha=args.alpha,beta=args.beta,double_dayu=args.double_dayu,smooth_thre= args.smooth_thre
                )

if __name__ == '__main__':
    main()
