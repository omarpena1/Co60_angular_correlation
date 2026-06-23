# src/co60/drs4.py
# Everything here is generic DRS4, no Co-60, no channel assumptions

from io import BufferedReader, FileIO
import struct
import numpy as np
from datetime import datetime
from collections import namedtuple

Event = namedtuple(
    'Event',
    [
        'event_id',
        'timestamp',
        'voltage_data',
        'time_data',
        'scalers',
    ]
)

class DRS4BinaryFile(BufferedReader):

    def __init__(self, filename):
        super().__init__(FileIO(filename, 'rb'))

        assert self.read(4) == b'DRS2', 'File does not seem to be a DRS4 binary file'
        assert self.read(4) == b'TIME', 'File does not contain TIME header'

        self.board_ids = []
        self.channels = {}
        self.time_widths = {}

        header = self.read(4)
        while header.startswith(b'B#'):
            board_id, = struct.unpack('H', header[2:])
            self.board_ids.append(board_id)
            self.time_widths[board_id] = {}
            self.channels[board_id] = []

            header = self.read(4)
            while header.startswith(b'C'):
                channel = int(header[1:].decode())
                self.channels[board_id].append(channel)

                self.time_widths[board_id][channel] =  self._read_timewidth_array()

                header = self.read(4)

        self.num_boards = len(self.board_ids)

        self.seek(-4, 1)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            header = self.read(4)
        except IOError:
            raise StopIteration

        if header != b'EHDR':
            raise StopIteration

        event_id, = struct.unpack('I', self.read(4))
        year, month, day, hour, minute, second, ms = struct.unpack(
            '7H', self.read(struct.calcsize('7H'))
        )
        timestamp = datetime(year, month, day, hour, minute, second, ms * 1000)
        range_center, = struct.unpack('H', self.read(2))

        scalers = {}
        trigger_cells = {}
        adc_data = {}
        voltage_data = {}
        time_data = {}

        for board_id, channels in self.channels.items():
            assert self.read(2) == b'B#'
            assert struct.unpack('H', self.read(2))[0] == board_id

            assert self.read(2) == b'T#'
            trigger_cells[board_id], = struct.unpack('H', self.read(2))

            scalers[board_id] = {}
            adc_data[board_id] = {}
            voltage_data[board_id] = {}
            time_data[board_id] = {}

            for channel in channels:
                assert self.read(4) == 'C{:03d}'.format(channel).encode('ascii')

                scalers[board_id][channel], = struct.unpack('I', self.read(4))
                adc_data[board_id][channel] = self._read_adc_data()
                voltage_data[board_id][channel] = [ (adc/(2**16) + range_center - 0.5) for adc in adc_data[board_id][channel]] #V
                time_data[board_id][channel] = self._calibrate_time(trigger_cells[board_id],self.time_widths[board_id][channel]) #ns

        return Event(
            event_id=event_id,
            timestamp=timestamp,
            voltage_data=voltage_data,
            scalers=scalers,
            time_data=time_data,
        )

    def _read_timewidth_array(self):
        return np.frombuffer(self.read(1024 * 4), 'float32')

    def _read_adc_data(self):
        return np.frombuffer(self.read(1024 * 2), 'uint16')

    def _calibrate_time(self,trigcell,twidths):
        t_calib = twidths
        # Make numpy array
        t_calib = np.array(t_calib)
        # Offset to trigger cell
        t_calib = np.roll(t_calib,trigcell)
        # Add zero time and remove last element
        t_calib = np.insert(t_calib, 0, 0., axis=0)
        t_calib = np.delete(t_calib,-1)
        # Compute calibrated sample times
        t_calib = np.cumsum(t_calib)
        
        return t_calib

# with DRS4BinaryFile('C:/Users/bmehrdel/Desktop/HybridPETdata/Hybrid TOF-PET-TwoBGO/19-04-2023/Na22-Single ended/Na22-Single ended/twoBGO-Na22-TG0.1V-output Voltage 42V-20April2023.dat') as f:
#     # print( len(f) )
#     # print(f.board_ids)
#     # print(f.channels)
#     # event = next(f)
   
    
#     events = [e for e in f]

def integrateWaveform(wf,ped_start=1,ped_end=50,int_start=60,int_end=1000):
    wf = np.array(wf)

    #Pedestal
    pedestal = wf[ped_start:ped_end].sum()/(ped_end-ped_start)
#     print(f"Pedestal:{pedestal}")
    
    #Integral
    signal_region = wf[int_start:int_end]
#     print(f"signal_region(before pedestal correction):{signal_region}")
    
    charge = signal_region.sum() - pedestal*(int_end-int_start)
#     print(f"Integrated charge:{charge}")
    
    return charge

def getPedestal(wf,ped_start=1,ped_end=50):
    wf = np.array(wf)

    #Pedestal
    pedestal = wf[ped_start:ped_end].sum()/(ped_end-ped_start)
     
    return pedestal

# def timeAtThreshold(algorithm, wf_time, wf, thres=0.005, ped_start=1, ped_end=50, debug=False):
def timeAtThreshold(algorithm, wf_time, wf, thres=0.02, ped_start=1, ped_end=50, debug=False, is_negative_polarity = False,):

    wf = np.array(wf)

    if is_negative_polarity:
        wf = -wf

    wf_time = np.array(wf_time)

    # Pedestal correction
    pedestal = wf[ped_start:ped_end].sum()/(ped_end-ped_start)
    #Pedestal stability
    if wf[ped_start:ped_end].std() > 0.01:
        return -99999. # pedestal not stable
    
    s_thres = 0 # Sample at threshold
    if algorithm == 0:
        # Find max and walk back
        s_max = np.argmax(wf)
        s_thres = np.where(wf[0:s_max]<thres)[0] # Collection of samples below threshold
        if len(s_thres) == 0:
            return -99999. # didn't cross threshold
        s_thres = s_thres[-1] # Last sample below threshold right before crossing

    if algorithm == 1:
       # Correct by pedestal
        thres += pedestal
        # Find max and walk back
        s_max = np.argmax(wf)
        s_thres = np.where(wf[0:s_max]<thres)[0] # Collection of samples below threshold
        if len(s_thres) == 0:
            return -99999. # didn't cross threshold
        s_thres = s_thres[-1] # Last sample below threshold

    elif algorithm == 2:
        # Find coarse threshold and walk back
        thres += pedestal
        thres_coarse = thres*20
        s_max = np.argmax(wf)
        s_thres_coarse = np.where(wf[0:s_max]<thres_coarse)[0] # Collection of samples below coarse threshold
        if len(s_thres_coarse) == 0:
            return -99999. # didn't cross coarse threshold
        s_thres_coarse = s_thres_coarse[-1] # Last sample below threshold right before crossing
        s_thres = np.where(wf[0:s_thres_coarse]<thres)[0] # Collection of samples below threshold before coarse threshold
        if len(s_thres) == 0:
            return -99999. # didn't cross threshold
        s_thres = s_thres[-1] # Last sample below threshold right before crossing

    elif algorithm == 3:
        # Correct by pedestal
        thres += pedestal
        # Find max
        s_max = np.argmax(wf)
        # Find first sample above threshold
        s_thress = np.where(wf[0:s_max]>thres)[0] # Collection of samples above threshold
        if len(s_thress) == 0:
            return -99999. # didn't cross threshold
        for s in s_thress:
            if ((wf[s] > thres) & (wf[s-1] < thres)): #Crossed threshold
                s_thres = s - 1 # First sample below threshold right before crossing
                break
          
    # Check if we actually crossed threshold
    if (wf[s_thres] < thres) & (wf[s_thres+1] > thres):
        # We good, interpolate
        t_thres = wf_time[s_thres] + (thres - wf[s_thres])/(wf[s_thres+1] - wf[s_thres])*(wf_time[s_thres+1] - wf_time[s_thres])
    else:
        return -99999. # didn't cross threshold

    # print('s_max',s_max,'s_thres',s_thres,'t_thres',t_thres)
    
    if debug:
        import matplotlib as plt
        plt.plot(wf_time,wf,color='C0',alpha=0.2)
        plt.plot(t_thres,thres,marker='o',color='Black')
        # plt.xlim(0,100)
        plt.ylim(-0.2,0.2)
    
    return t_thres


def readDRS4Binary(fname, nevents=None, is_negative_polarity=False):
    """
    Iterate over events in a DRS4 file, up to at most `nevents`.
    If nevents is None, read the whole file.
    """
    with DRS4BinaryFile(fname) as f:
        for i, event in enumerate(f):
            if (nevents is not None) and (i >= nevents):
                break

            results = {}  # Boards:Channels:Observables

            for board_id in f.board_ids:
                board_results = {}  # Channels:Observables

                for channel in f.channels[board_id]:
                    channel_results = {}

                    energy = integrateWaveform(event.voltage_data[board_id][channel])
                    thres = timeAtThreshold(
                        1,
                        event.time_data[board_id][channel],
                        event.voltage_data[board_id][channel],
                        is_negative_polarity= is_negative_polarity
                    )
                    voltages = event.voltage_data[board_id][channel]
                    times = event.time_data[board_id][channel]
                    pedestal = getPedestal(event.voltage_data[board_id][channel])

                    channel_results["Pedestal"] = pedestal
                    channel_results["Voltages"] = voltages
                    channel_results["Times"] = times
                    channel_results["Energy"] = energy
                    channel_results["Tthres"] = thres

                    board_results[channel] = channel_results

                results[board_id] = board_results

            yield results