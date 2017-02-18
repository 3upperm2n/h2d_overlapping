#!/usr/bin/env python

"""
Check the timing for overlapping transfer on GPU device.
"""
import warnings
import subprocess
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class Transfer(object):
    """
    class to store star and end time in ms
    """
    def __init__(self, start=0.0, end=0.0):
        self.start_time_ms = start
        self.end_time_ms = end


class Streams(object):
    """
    class to store CUDA stream api timing
    """
    def __init__(self):
        self.h2d = []
        self.d2h = []
        self.kernel = []


def time_coef_ms(df_trace):
    """
    check the timing unit and using ms by adjusting the coefficients
    """
    start_unit = df_trace['Start'].iloc[0]
    duration_unit = df_trace['Duration'].iloc[0]

    start_coef = 1.0
    if start_unit == 's':
        start_coef = 1e3
    if start_unit == 'us':
        start_coef = 1e-3

    duration_coef = 1.0
    if duration_unit == 's':
        duration_coef = 1e3
    if duration_unit == 'us':
        duration_coef = 1e-3

    return start_coef, duration_coef


def read_row(df_row, start_coef_ms, duration_coef_ms):
    """
    Read one row from the dataframe and extract api call info.
    """
    start_time_ms = float(df_row['Start']) * start_coef_ms

    end_time_ms = start_time_ms + float(df_row['Duration']) * duration_coef_ms

    stream_id = int(df_row['Stream'])

    api_name = df_row['Name'].to_string()

    if "DtoH" in api_name:
        api_type = 'd2h'
    elif "HtoD" in api_name:
        api_type = 'h2d'
    else:
        api_type = 'kernel'

    return stream_id, api_type, start_time_ms, end_time_ms


def read_trace(trace_file):
    """
    Read trace in csv using pandas.
    """
    # There are max 17 columns in the output csv
    col_name = ["Start", "Duration", "Grid X", "Grid Y", "Grid Z", "Block X",
                "Block Y", "Block Z", "Registers Per Thread", "Static SMem",
                "Dynamic SMem", "Size", "Throughput", "Device", "Context",
                "Stream", "Name"]

    df_trace = pd.read_csv(trace_file, names=col_name, engine='python')

    rows_to_skip = 0

    # find out the number of rows to skip
    for index, row in df_trace.iterrows():
        if row['Start'] == 'Start':
            rows_to_skip = index
            break

    # read the input csv again
    df_trace = pd.read_csv(trace_file, skiprows=rows_to_skip)

    return df_trace


def set_up_streams(df_trace):
    """
    Configure the stream list
    """
    # read the number of unique streams
    stream_id_list = df_trace['Stream'].unique()
    stream_id_list = stream_id_list[~np.isnan(stream_id_list)]  # remove nan
    num_streams = len(stream_id_list)

    streamlist = [None] * num_streams
    for i in range(num_streams):
        streamlist[i] = Streams()

    return streamlist, stream_id_list


def update_stream(df_trace, streamlist, stream_id_list, start_coef,
                  duration_coef):
    """
    record stream info from the datatrace
    """
    # read row by row
    for rowid in range(1, df_trace.shape[0]):
        #  extract info from the current row
        stream_id, api_type, start_time_ms, end_time_ms = \
            read_row(df_trace.iloc[[rowid]], start_coef, duration_coef)

        # find out index of the stream
        sid, = np.where(stream_id_list == stream_id)

        # add the start/end time for different api calls
        if api_type == 'h2d':
            streamlist[sid].h2d.append(Transfer(start_time_ms, end_time_ms))
        elif api_type == 'd2h':
            streamlist[sid].d2h.append(Transfer(start_time_ms, end_time_ms))
        elif api_type == 'kernel':
            streamlist[sid].kernel.append(Transfer(start_time_ms, end_time_ms))
        else:
            print "Unknown. Error."

    return streamlist


def check_overlap(streamlist):
    """
    check whether two cuda streams has data transfer overlap
    """
    h2d_api_list = []
    d2h_api_list = []

    #for i in range(len(streamlist)):
    for _, current_stream in enumerate(streamlist):
        h2d_ = Transfer(current_stream.h2d[0].start_time_ms,
                        current_stream.h2d[0].end_time_ms)

        for j in range(1, len(current_stream.h2d)):
            if h2d_.start_time_ms > current_stream.h2d[j].start_time_ms:
                h2d_.start_time_ms = current_stream.h2d[j].start_time_ms

            if h2d_.end_time_ms < current_stream.h2d[j].end_time_ms:
                h2d_.end_time_ms = current_stream.h2d[j].end_time_ms

        # append the h2d timing for the current stream
        h2d_api_list.append(h2d_)

        d2h_ = Transfer(current_stream.d2h[0].start_time_ms,
                        current_stream.d2h[0].end_time_ms)

        for j in range(1, len(current_stream.d2h)):
            if d2h_.start_time_ms > current_stream.d2h[j].start_time_ms:
                d2h_.start_time_ms = current_stream.d2h[j].start_time_ms

            if d2h_.end_time_ms < current_stream.d2h[j].end_time_ms:
                d2h_.end_time_ms = current_stream.d2h[j].end_time_ms

        # append the d2h timing for the current stream
        d2h_api_list.append(d2h_)

    # --------
    # check H2D-H2D Overapping
    # --------
    stream_0_h2d_start = h2d_api_list[0].start_time_ms
    stream_0_h2d_end = h2d_api_list[0].end_time_ms

    h2d_h2d_ovlp = 0
    h2h_h2d_ovhd_ms = 0.0

    for i in range(1, len(streamlist)):
        prev_stream_api = h2d_api_list[i-1].start_time_ms
        current_stream_api = h2d_api_list[i].start_time_ms

        # h2d launch overhead in ms
        h2h_h2d_ovhd_ms += current_stream_api - prev_stream_api

        if current_stream_api >= stream_0_h2d_start and \
                current_stream_api <= stream_0_h2d_end:
            h2d_h2d_ovlp = 1

    # compute the avg ovhd
    h2d_h2d_ovhd_ms = h2h_h2d_ovhd_ms / float(len(streamlist) - 1)

    # --------
    # check D2H-H2D Overapping
    # --------
    stream_0_d2h_start = d2h_api_list[0].start_time_ms

    d2h_h2d_ovlp = 0
    d2h_h2d_ovhd_ms = 0.0

    for i in range(1, len(streamlist)):
        pre_h2d_start = h2d_api_list[i-1].start_time_ms

        cur_h2d_start = h2d_api_list[i].start_time_ms
        cur_h2d_end = h2d_api_list[i].end_time_ms

        # h2d launch overhead in ms
        d2h_h2d_ovhd_ms += cur_h2d_start - pre_h2d_start

        if stream_0_d2h_start >= cur_h2d_start and \
                stream_0_d2h_start <= cur_h2d_end:
            d2h_h2d_ovlp = 1

    # compute the avg ovhd
    d2h_h2d_ovhd_ms = d2h_h2d_ovhd_ms / float(len(streamlist) - 1)

    return d2h_h2d_ovlp, d2h_h2d_ovhd_ms, h2d_h2d_ovlp, h2d_h2d_ovhd_ms


def run_trace(input_num):
    """
    Generate trace and check the overlapping
    """
    # ----------------
    # call shell script to generate the trace
    # ----------------
    input_num_str = str(input_num)
    subprocess.call(['./1_gen_trace.sh', input_num_str])

    # ----------------
    # read trace
    # ----------------
    df_trace = read_trace('trace.csv')

    # ----------------
    # set up the stream list
    # ----------------
    streamlist, stream_id_list = set_up_streams(df_trace)

    # ----------------
    # adjust the timing unit to ms
    # ----------------
    start_coef, duration_coef = time_coef_ms(df_trace)

    # ----------------
    # Read data frame row by row, to update the streamList[].
    # ----------------
    streamlist = update_stream(df_trace, streamlist, stream_id_list, start_coef,
                               duration_coef)

    # ----------------
    # check d2h-h2d / h2d-h2d overlap
    # ----------------
    d2h_h2d_ovlp, d2h_h2d_ovhd_ms, h2d_h2d_ovlp, h2d_h2d_ovhd_ms = \
        check_overlap(streamlist)

    return d2h_h2d_ovlp, d2h_h2d_ovhd_ms, h2d_h2d_ovlp, h2d_h2d_ovhd_ms


def main():
    """
    Main Function.
    """

    input_num = 10000  # start from 10K floats

    quiet_d2h = 0

    while True:
        d2h_h2d_ovlp, d2h_h2d_ovhd_ms, h2d_h2d_ovlp, h2d_h2d_ovhd_ms = \
            run_trace(input_num)

        if d2h_h2d_ovlp == 1 and h2d_h2d_ovlp == 0 and quiet_d2h == 0:
            print "N = %d, d2h-h2d overlap : %f (ms)\n" % (input_num, d2h_h2d_ovhd_ms)
            quiet_d2h = 1

        if h2d_h2d_ovlp == 1:
            print "N = %d, h2d-h2d overlap : %f (ms)\n" % (input_num, h2d_h2d_ovhd_ms)
            break

        input_num += 100  # increment 100 floats


if __name__ == "__main__":
    main()
