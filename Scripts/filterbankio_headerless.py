"""
split off from filterbank.py in https://github.com/UCBerkeleySETI/filterbank

This provides a class, Filterbank(), which can be used to read a .fil file:

TODO: add support for .tim time series files

"""

import numpy as np
import struct
import os
from astropy import units as u
from astropy.coordinates import Angle
import re

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

MAX_DATA_ARRAY_SIZE = 1024 * 1024 * 1024     # Max size of data array to load into memory
MAX_HEADER_BLOCKS   = 100                    # Max size of header (in 512-byte blocks)

###
# Header parsing
###

# Dictionary of allowed keywords and their types
# Here are the keywordss that a filter bank file may
# contain.  Items marked with "[*]" are not yet # supported.  See docs for
# indivisuabl attribtues for more detailed info.
#
#   * telescope_id (int): 0=fake data; 1=Arecibo; 2=Ooty... others to be added
#   * machine_id (int): 0=FAKE; 1=PSPM; 2=WAPP; 3=OOTY... others to be added
#   * data_type (int): 1=filterbank; 2=time series... others to be added
#   * rawdatafile (string): the name of the original data file
#   * source_name (string): the name of the source being observed by the telescope
#   * barycentric (int): equals 1 if data are barycentric or 0 otherwise
#   * pulsarcentric (int): equals 1 if data are pulsarcentric or 0 otherwise
#   * az_start (double): telescope azimuth at start of scan (degrees)
#   * za_start (double): telescope zenith angle at start of scan (degrees)
#   * src_raj (double): right ascension (J2000) of source (hours, converted from hhmmss.s)
#   * src_dej (double): declination (J2000) of source (degrees, converted from ddmmss.s)
#   * tstart (double): time stamp (MJD) of first sample
#   * tsamp (double): time interval between samples (s)
#   * nbits (int): number of bits per time sample
#   * nsamples (int): number of time samples in the data file (rarely used any more)
#   * fch1 (double): centre frequency (MHz) of first filterbank channel
#   * foff (double): filterbank channel bandwidth (MHz)
#   * FREQUENCY_START [*] (character): start of frequency table (see below for explanation)
#   * fchannel [*] (double): frequency channel value (MHz)
#   * FREQUENCY_END [*] (character): end of frequency table (see below for explanation)
#   * nchans (int): number of filterbank channels
#   * nifs (int): number of seperate IF channels
#   * refdm (double): reference dispersion measure (pc/cm**3)
#   * period (double): folding period (s)
#   * nbeams (int):total number of beams (?)
#   * ibeam (int): number of the beam in this file (?)

header_keyword_types = {
    b'telescope_id' : b'<l',
    b'machine_id'   : b'<l',
    b'data_type'    : b'<l',
    b'barycentric'  : b'<l',
    b'pulsarcentric': b'<l',
    b'nbits'        : b'<l',
    b'nsamples'     : b'<l',
    b'nchans'       : b'<l',
    b'nifs'         : b'<l',
    b'nbeams'       : b'<l',
    b'ibeam'        : b'<l',
    b'rawdatafile'  : b'str',
    b'source_name'  : b'str',
    b'az_start'     : b'<d',
    b'za_start'     : b'<d',
    b'tstart'       : b'<d',
    b'tsamp'        : b'<d',
    b'fch1'         : b'<d',
    b'foff'         : b'<d',
    b'refdm'        : b'<d',
    b'period'       : b'<d',
    b'src_raj'      : b'angle',
    b'src_dej'      : b'angle',
    }

def grab_header(filename):
    """ Extract the filterbank header from the file

    Args:
        filename (str): name of file to open

    Returns:
        header_str (str): filterbank header as a binary string
    """
    f = open(filename, 'rb')
    eoh_found = False

    header_str = ''
    header_sub_count = 0
    while not eoh_found:
        header_sub = f.read(512)
        header_sub_count += 1
        if 'HEADER_START' in header_sub:
            idx_start = header_sub.index('HEADER_START') + len('HEADER_START')
            header_sub = header_sub[idx_start:]

        if 'HEADER_END' in header_sub:
            eoh_found = True
            idx_end = header_sub.index('HEADER_END')
            header_sub = header_sub[:idx_end]

        if header_sub_count >= MAX_HEADER_BLOCKS:
            raise RuntimeError("MAX HEADER LENGTH REACHED. THIS FILE IS FUBARRED.")
        header_str += header_sub

    f.close()
    return header_str

def len_header(filename):
    """ Return the length of the filterbank header, in bytes

    Args:
        filename (str): name of file to open

    Returns:
        idx_end (int): length of header, in bytes
    """
    with  open(filename, 'rb') as f:
        header_sub_count = 0
        eoh_found = False
        while not eoh_found:
            header_sub = f.read(512)
            header_sub_count += 1
            if b'HEADER_END' in header_sub:
                idx_end = header_sub.index(b'HEADER_END') + len(b'HEADER_END')
                eoh_found = True
                break

        idx_end = (header_sub_count -1) * 512 + idx_end
    return idx_end

def parse_header(filename):
    """ Parse a header of a filterbank, looking for allowed keywords

    Uses header_keyword_types dictionary as a lookup for data types.

    Args:
        filename (str): name of file to open

    Returns:
        header_dict (dict): A dictioary of header key:value pairs
    """
    header = grab_header(filename)
    header_dict = {}

    #print header
    for keyword in header_keyword_types.keys():
        if keyword in header:
            dtype = header_keyword_types.get(keyword, 'str')
            idx = header.index(keyword) + len(keyword)
            dtype = header_keyword_types[keyword]
            if dtype == '<l':
                val = struct.unpack(dtype, header[idx:idx+4])[0]
                header_dict[keyword] = val
            if dtype == '<d':
                val = struct.unpack(dtype, header[idx:idx+8])[0]
                header_dict[keyword] = val
            if dtype == 'str':
                str_len = struct.unpack('<L', header[idx:idx+4])[0]
                str_val = header[idx+4:idx+4+str_len]
                header_dict[keyword] = str_val
            if dtype == 'angle':
                val = struct.unpack('<d', header[idx:idx+8])[0]
                val = fil_double_to_angle(val)

                if keyword == 'src_raj':
                    val = Angle(val, unit=u.hour)
                else:
                    val = Angle(val, unit=u.deg)
                header_dict[keyword] = val

    return header_dict

def read_next_header_keyword(fh):
    """

    Args:
        fh (file): file handler

    Returns:
    """
    n_bytes = np.fromstring(fh.read(4), dtype='uint32')[0]
    #print n_bytes

    if n_bytes > 255:
        n_bytes = 16

    keyword = fh.read(n_bytes)

    #print keyword

    if keyword == b'HEADER_START' or keyword == b'HEADER_END':
        return keyword, 0, fh.tell()
    else:
        dtype = header_keyword_types[keyword]
        #print dtype
        idx = fh.tell()
        if dtype == b'<l':
            val = struct.unpack(dtype, fh.read(4))[0]
        if dtype == b'<d':
            val = struct.unpack(dtype, fh.read(8))[0]
        if dtype == b'str':
            str_len = np.fromstring(fh.read(4), dtype='int32')[0]
            val = fh.read(str_len)
        if dtype == b'angle':
            val = struct.unpack('<d', fh.read(8))[0]
            val = fil_double_to_angle(val)
            if keyword == b'src_raj':
                val = Angle(val, unit=u.hour)
            else:
                val = Angle(val, unit=u.deg)
        return keyword, val, idx

def read_header(filename, return_idxs=False):
    """ Read filterbank header and return a Python dictionary of key:value pairs

    Args:
        filename (str): name of file to open

    Optional args:
        return_idxs (bool): Default False. If true, returns the file offset indexes
                            for values

    returns

    """
    with open(filename, 'rb') as fh:
        header_dict = {}
        header_idxs = {}

        # Check this is a filterbank file
        keyword, value, idx = read_next_header_keyword(fh)

        try:
            assert keyword == b'HEADER_START'
        except AssertionError:
            raise RuntimeError("Not a valid filterbank file.")

        while True:
            keyword, value, idx = read_next_header_keyword(fh)
            if keyword == b'HEADER_END':
                break
            else:
                header_dict[keyword] = value
                header_idxs[keyword] = idx

    if return_idxs:
        return header_idxs
    else:
        return header_dict

def fix_header(filename, keyword, new_value):
    """ Apply a quick patch-up to a Filterbank header by overwriting a header value


    Args:
        filename (str): name of file to open and fix. WILL BE MODIFIED.
        keyword (stt):  header keyword to update
        new_value (long, double, angle or string): New value to write.

    Notes:
        This will overwrite the current value of the filterbank with a desired
        'fixed' version. Note that this has limited support for patching
        string-type values - if the length of the string changes, all hell will
        break loose.

    """

    # Read header data and return indexes of data offsets in file
    hd = read_header(filename)
    hi = read_header(filename, return_idxs=True)
    idx = hi[keyword]

    # Find out the datatype for the given keyword
    dtype = header_keyword_types[keyword]
    dtype_to_type = {'<l'  : np.int32,
                     'str' : str,
                     '<d'  : np.float64,
                     'angle' : to_sigproc_angle}
    value_dtype = dtype_to_type[dtype]

    # Generate the new string
    if value_dtype is str:
        if len(hd[keyword]) == len(new_value):
            val_str = np.int32(len(new_value)).tostring() + new_value
        else:
            raise RuntimeError("String size mismatch. Cannot update without rewriting entire file.")
    else:
        val_str = value_dtype(new_value).tostring()

    # Write the new string to file
    with open(filename, 'rb+') as fh:
        fh.seek(idx)
        fh.write(val_str)

def fil_double_to_angle(angle):
      """ Reads a little-endian double in ddmmss.s (or hhmmss.s) format and then
      converts to Float degrees (or hours).  This is primarily used to read
      src_raj and src_dej header values. """

      negative = (angle < 0.0)
      angle = np.abs(angle)

      dd = np.floor((angle / 10000))
      angle -= 10000 * dd
      mm = np.floor((angle / 100))
      ss = angle - 100 * mm
      dd += mm/60.0 + ss/3600.0

      if negative:
          dd *= -1

      return dd

###
# sigproc writing functions
###

def to_sigproc_keyword(keyword, value=None):
    """ Generate a serialized string for a sigproc keyword:value pair

    If value=None, just the keyword will be written with no payload.
    Data type is inferred by keyword name (via a lookup table)

    Args:
        keyword (str): Keyword to write
        value (None, float, str, double or angle): value to write to file

    Returns:
        value_str (str): serialized string to write to file.
    """

    keyword = str(keyword)

    if not value:
        return np.int32(len(keyword)).tostring() + keyword
    else:
        dtype = header_keyword_types[keyword]

        dtype_to_type = {'<l'  : np.int32,
                         'str' : str,
                         '<d'  : np.float64,
                         'angle' : to_sigproc_angle}

        value_dtype = dtype_to_type[dtype]

        if value_dtype is str:
            return np.int32(len(keyword)).tostring() + keyword + np.int32(len(value)).tostring() + value
        else:
            return np.int32(len(keyword)).tostring() + keyword + value_dtype(value).tostring()

def generate_sigproc_header(f):
    """ Generate a serialzed sigproc header which can be written to disk.

    Args:
        f (Filterbank object): Filterbank object for which to generate header

    Returns:
        header_str (str): Serialized string corresponding to header
    """

    header_string = ''
    header_string += to_sigproc_keyword('HEADER_START')

    for keyword in f.header.keys():
        if keyword == 'src_raj':
            header_string += to_sigproc_keyword('src_raj')  + to_sigproc_angle(f.header['src_raj'])
        elif keyword == 'src_dej':
            header_string += to_sigproc_keyword('src_dej')  + to_sigproc_angle(f.header['src_dej'])
        elif keyword == 'az_start' or keyword == 'za_start':
            header_string += to_sigproc_keyword(keyword)  + np.float64(f.header[keyword]).tostring()
        elif keyword not in header_keyword_types.keys():
            pass
        else:
            header_string += to_sigproc_keyword(keyword, f.header[keyword])

    header_string += to_sigproc_keyword('HEADER_END')
    return header_string

def to_sigproc_angle(angle_val):
    """ Convert an astropy.Angle to the ridiculous sigproc angle format string. """
    x = str(angle_val)

    if 'h' in x:
        d, m, s, ss = int(x[0:x.index('h')]), int(x[x.index('h')+1:x.index('m')]), \
        int(x[x.index('m')+1:x.index('.')]), float(x[x.index('.'):x.index('s')])
    if 'd' in x:
        d, m, s, ss = int(x[0:x.index('d')]), int(x[x.index('d')+1:x.index('m')]), \
        int(x[x.index('m')+1:x.index('.')]), float(x[x.index('.'):x.index('s')])
    num = str(d).zfill(2) + str(m).zfill(2) + str(s).zfill(2)+ '.' + str(ss).split(".")[-1]
    return np.float64(num).tostring()

###
# Main filterbank class
###

class Filterbank(object):
    """ Class for loading and plotting filterbank data """

    def __repr__(self):
        return "Filterbank data: %s" % self.filename

    def __init__(self, filename, Block_Number, f_start=None, f_stop=None,
                 t_start=None, t_stop=None, load_data=True,
                 header_dict=None, data_array=None, my_header=None):
        """ Class for loading and plotting filterbank data.

        This class parses the filterbank file and stores the header and data
        as objects:
            fb = Filterbank('filename_here.fil')
            fb.header        # filterbank header, as a dictionary
            fb.data          # filterbank data, as a numpy array

        Args:
            filename (str): filename of filterbank file.
            Block_Number (int): number of block we wish to inspect
            f_start (float): start frequency in MHz
            f_stop (float): stop frequency in MHz
            t_start (int): start integration ID
            t_stop (int): stop integration ID
            load_data (bool): load data. If set to False, only header will be read.
            header_dict (dict): Create filterbank from header dictionary + data array
            data_array (np.array): Create filterbank from header dict + data array
            my_header (list): my_header has form [f0,f_delt,nbytes,nchans,nifs,tsamp]
           
        """
        
     
        
        # Check if block is in odd or even beam
        filename_numbers = re.findall(r'\d+',filename)
        beam_number = int(filename_numbers[0])

        # my_header has form [f0,f_delt,n_bytes,n_chans,n_ifs,tsamp]

        if beam_number%2 == 0:
            my_header = [148.9, -0.00305176, 4, 1920, 1, 327.68e-6]     # Even beam numbers
        
        elif beam_number%2 == 1:
            my_header = [148.9, -0.00305176, 4, 1984, 1, 327.68e-6]     # Odd beam numbers
        
        if filename:
            self.filename = filename
            if HAS_HDF5:
                if h5py.is_hdf5(filename):
                    self.read_hdf5(filename, f_start, f_stop, t_start, t_stop, load_data)
                else:
                    self.read_filterbank(filename, Block_Number, f_start, f_stop, t_start, t_stop, load_data, my_header)
            else:
                self.read_filterbank(filename, Block_Number, f_start, f_stop, t_start, t_stop, load_data, my_header)
        elif header_dict is not None and data_array is not None: 
            self.gen_from_header(header_dict, data_array)
        else:
            pass

    def gen_from_header(self, header_dict, data_array, f_start=None, f_stop=None,
                        t_start=None, t_stop=None, load_data=True):
        self.filename = ''
        self.header = header_dict
        self.data = data_array
        self.n_ints_in_file = 0

        self._setup_freqs()

    def read_hdf5(self, filename, f_start=None, f_stop=None,
                        t_start=None, t_stop=None, load_data=True):
        self.header = {}
        self.filename = filename
        self.h5 = h5py.File(filename)
        for key, val in self.h5['data'].attrs.items():
            if key == b'src_raj':
                self.header[key] = Angle(val, unit='hr')
            elif key == b'src_dej':
                self.header[key] = Angle(val, unit='deg')
            else:
                self.header[key] = val

        self.data = self.h5["data"][:]
        self._setup_freqs()

        self.n_ints_in_file  = self.data.shape[0]
        self.file_size_bytes = os.path.getsize(self.filename)

    def _setup_freqs(self, f_start=None, f_stop=None):
        ## Setup frequency axis        
        f0 = self.my_header[0]
        f_delt = self.my_header[1]
        
        #i_start, i_stop = 0, self.header[b'nchans']
        i_start, i_stop = 0, self.my_header[3]
        if f_start:
            i_start = (f_start - f0) / f_delt
        if f_stop:
            i_stop  = (f_stop - f0)  / f_delt

        #calculate closest true index value
        chan_start_idx = np.int(i_start)
        chan_stop_idx  = np.int(i_stop)

        #create freq array
        if i_start < i_stop:
            i_vals = np.arange(chan_start_idx, chan_stop_idx)
        else:
            i_vals = np.arange(chan_stop_idx, chan_start_idx)

        self.freqs = f_delt * i_vals + f0

        if f_delt < 0:
            self.freqs = self.freqs[::-1]

        return i_start, i_stop, chan_start_idx, chan_stop_idx

    def read_filterbank(self, filename, Block_Number, f_start=None, f_stop=None,
                        t_start=None, t_stop=None, load_data=True, my_header=None):
        
        
        if filename is None:
            filename = self.filename
            
        if my_header:
            [f0,f_delt,n_bytes,n_chans,n_ifs,t_delt] = my_header     # If file has no header use the values defined in my_header
            self.my_header = my_header
            self.block_number = Block_Number
            t_start = (self.block_number - 1)*32768
            if t_start:
                t_stop = t_start + 32768     
            else:
                t_stop = 32768
        else:
            self.header = read_header(filename)

            # Setup frequency axis
            f0 = self.header[b'fch1']
            f_delt = self.header[b'foff']
            n_bytes  = self.header[b'nbits'] / 8
            n_chans = self.header[b'nchans']
            n_ifs   = self.header[b'nifs']
            t_delt = self.header[b'tsamp']
            self.my_header = [f0,f_delt,n_bytes,n_chans,n_ifs,t_delt]
            
        
                         
        #convert input frequencies into what their corresponding index would be
        i_start, i_stop, chan_start_idx, chan_stop_idx = self._setup_freqs(f_start, f_stop)    
        n_chans_selected = self.freqs.shape[0] 
            
        # Load binary data
        #self.idx_data = len_header(filename)
        f = open(filename, 'rb')
        #f.seek(self.idx_data)
        #filesize = os.path.getsize(self.filename)
        #n_bytes_data = filesize - self.idx_data
        #n_ints_in_file = n_bytes_data / (n_bytes * n_chans * n_ifs)

        # now check to see how many integrations requested
        #ii_start, ii_stop = 0, n_ints_in_file
        ii_start = 0
        if t_start:
            ii_start = t_start
        if t_stop:
            ii_stop = t_stop
        n_ints = ii_stop - ii_start
       

        # Seek to first integration
        f.seek(int(ii_start * n_bytes * n_ifs * n_chans), 1)

        # Set up indexes used in file read (taken out of loop for speed)
        i0 = np.min((chan_start_idx, chan_stop_idx))
        i1 = np.max((chan_start_idx, chan_stop_idx))

        #Set up the data type (taken out of loop for speed)
        if n_bytes == 4:
            dd_type = 'float32'
        elif n_bytes == 2:
            dd_type = 'int16'
        elif n_bytes == 1:
            dd_type = 'int8'

        if load_data:

            if n_ints * n_ifs * n_chans_selected > MAX_DATA_ARRAY_SIZE:
                print("Error: data array is too large to load. Either select fewer")
                print("points or manually increase MAX_DATA_ARRAY_SIZE.")
                exit()

            
            self.data = np.zeros((int(n_ints), int(n_ifs), int(n_chans_selected)), dtype='float32')

            for ii in range(int(n_ints)):
                """d = f.read(n_bytes * n_chans * n_ifs)
                """

                for jj in range(n_ifs):

                    f.seek(int(n_bytes * i0), 1) # 1 = from current location
                    #d = f.read(n_bytes * n_chans_selected)
                    #bytes_to_read = n_bytes * n_chans_selected

                    dd = np.fromfile(f, count=n_chans_selected, dtype=dd_type)

                    # Reverse array if frequency axis is flipped
                    if f_delt < 0:
                        dd = dd[::-1]

                    self.data[ii, jj] = dd

                    f.seek(int(n_bytes * (n_chans - i1)), 1)  # Seek to start of next block
        else:
            print("Skipping data load...")
            self.data = np.array([0])

        # Finally add some other info to the class as objects
        #self.n_ints_in_file  = n_ints_in_file
        #self.file_size_bytes = filesize

        ## Setup time axis      
        #t0 = self.header[b'tstart'] 
        #if not t0:
        #    t0 = (block_number - 1)*32768
        #self.timestamps = np.arange(0, n_ints) * t_delt / 24./60./60 + t0

    def info(self):
        """ Print header information """

        for key, val in self.header.items():
            if key == 'src_raj':
                val = val.to_string(unit=u.hour, sep=':')
            if key == 'src_dej':
                val = val.to_string(unit=u.deg, sep=':')
            print("%16s : %32s" % (key, val))

        print("\n%16s : %32s" % ("Num ints in file", self.n_ints_in_file))
        print("%16s : %32s" % ("Data shape", self.data.shape))
        print("%16s : %32s" % ("Start freq (MHz)", self.freqs[0]))
        print("%16s : %32s" % ("Stop freq (MHz)", self.freqs[-1]))

    def generate_freqs(self, f_start, f_stop):
        """
        returns frequency array [f_start...f_stop]
        """

        fch1 = self.header[b'fch1']
        foff = self.header[b'foff']

        #convert input frequencies into what their corresponding index would be
        i_start = (f_start - fch1) / foff
        i_stop  = (f_stop - fch1)  / foff

        #calculate closest true index value
        chan_start_idx = np.int(i_start)
        chan_stop_idx  = np.int(i_stop)

        #create freq array
        i_vals = np.arange(chan_stop_idx, chan_start_idx, 1)

        freqs = foff * i_vals + fch1

        return freqs[::-1]

    def grab_data(self, f_start=None, f_stop=None, if_id=0):
        """ Extract a portion of data by frequency range.

        Args:
            f_start (float): start frequency in MHz
            f_stop (float): stop frequency in MHz
            if_id (int): IF input identification (req. when multiple IFs in file)

        Returns:
            (freqs, data) (np.arrays): frequency axis in MHz and data subset
        """
        i_start, i_stop = 0, None

        if f_start:
            i_start = closest(self.freqs, f_start)
        if f_stop:
            i_stop = closest(self.freqs, f_stop)

        plot_f    = self.freqs[i_start:i_stop]
        plot_data = self.data[:, if_id, i_start:i_stop]
        return plot_f, plot_data

    def write_to_filterbank(self, filename_out):
        """ Write data to filterbank file.

        Args:
            filename_out (str): Name of output file
        """

        #calibrate data
        #self.data = calibrate(mask(self.data.mean(axis=0)[0]))
        #rewrite header to be consistent with modified data
        self.header[b'fch1']   = self.freqs[0]
        self.header[b'foff']   = self.freqs[1] - self.freqs[0]
        self.header[b'nchans'] = self.freqs.shape[0]
        #self.header['tsamp']  = self.data.shape[0] * self.header['tsamp']

        n_bytes  = self.header[b'nbits'] / 8
        with open(filename_out, "w") as fileh:
            fileh.write(generate_sigproc_header(self))
            j = self.data
            if n_bytes == 4:
                np.float32(j[:, ::-1].ravel()).tofile(fileh)
            elif n_bytes == 2:
                np.int16(j[:, ::-1].ravel()).tofile(fileh)
            elif n_bytes == 1:
                np.int8(j[:, ::-1].ravel()).tofile(fileh)
