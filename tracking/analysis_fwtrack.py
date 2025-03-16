import matplotlib.pyplot as plt
import argparse
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import print_results_per_video
from lib.test.evaluation import get_dataset, trackerlist

def run_analyse(tracker_name,tracker_param, dataset_name, test_checkpoint = None):
    trackers = []
    
    if tracker_name == 'fwtrack':
        trackers.extend(trackerlist(name='fwtrack', parameter_name=tracker_param, dataset_name=dataset_name,
                                run_ids=None, test_checkpoint = test_checkpoint, display_name='FWTrack_256'))
        
    dataset = get_dataset(dataset_name)

    print_results_per_video(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),per_video=True)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--test_checkpoint', type=str, default=None)

    args = parser.parse_args()

    run_analyse(args.tracker_name,  args.tracker_param ,args.dataset_name, test_checkpoint = args.test_checkpoint)


if __name__ == '__main__':
    main()



