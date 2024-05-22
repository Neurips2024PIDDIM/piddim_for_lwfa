import os
import subprocess
import sys
import csv

def calculate_fid_statistics(dir1, dir2, start=1, end=22):
    sum_fid = 0
    sum_fid_sq = 0
    count = 0
    max_fid = -float('inf')
    min_fid = float('inf')
    max_subdir = ""
    min_subdir = ""

    for i in range(start, end + 1):
        subdir1 = os.path.join(dir1, str(i))
        subdir2 = os.path.join(dir2, str(i))

        if os.path.isdir(subdir1) and os.path.isdir(subdir2):
            output = subprocess.check_output(['python', '-m', 'pytorch_fid', '--device', 'cuda:1', subdir1, subdir2])
            output_str = output.decode('utf-8').strip()
            fid = float(output_str.split()[-1])

            sum_fid += fid
            sum_fid_sq += fid ** 2

            if fid > max_fid:
                max_fid = fid
                max_subdir = str(i)

            if fid < min_fid:
                min_fid = fid
                min_subdir = str(i)

            count += 1

    if count > 0:
        average_fid = sum_fid / count
        variance_fid = (sum_fid_sq - (sum_fid ** 2) / count) / (count - 1) if count > 1 else 0
        physics = dir2.split('_')[1]
        sections = dir2.split('_')[3]
        cfg = dir2.split('_cfg')[-1]

        return {
            'physics': physics,
            'sections': sections,
            'cfg': cfg,
            'average_fid': average_fid,
            'max_fid': max_fid,
            'min_fid': min_fid,
            'max_subdir': max_subdir,
            'min_subdir': min_subdir,
            'variance_fid': variance_fid
        }
    else:
        return None

if __name__ == "__main__":
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    with open('fid_statistics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Physics', 'Sections', 'CFG', 'Average FID', 'Maximum FID', 'Minimum FID', 'Max Subdirectory', 'Min Subdirectory', 'Variance of FID'])
    for dir in os.listdir(dir2):
        results = calculate_fid_statistics(dir1, dir2 + '/' + dir)

        if results:
            with open('fid_statistics.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([results['physics'], results['sections'], results['cfg'], results['average_fid'], results['max_fid'], results['min_fid'], results['max_subdir'], results['min_subdir'], results['variance_fid']])
