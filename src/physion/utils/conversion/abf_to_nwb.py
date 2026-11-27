"""
Convert spontaneous two-photon recordings (Bruker xml + abf rotary encoder
+ FaceIt + suite2p) to NWB.

Usage:

    python -m physion.utils.conversion.abf_to_nwb DataTable.xlsx \\
        --raw-root /raw/two-photon \\
        --processed-root /projects/<PRJ>/processed-data \\
        [--dataset <Dataset>]

DataTable.xlsx:

  - sheet "Recordings", one row per session, columns:
        session       (required) YYYY_MM_DD_HH-MM-SS
        subject       (required) key into the "Subjects" sheet
        dataset       (optional) raw dataset folder, defaults to --dataset
        protocol      (required) spontaneous / visual
                      (only "spontaneous" rows are converted here)
        experimenter  (required)
        include       (required) boolean, rows set to False are skipped
        comment       (optional)

  - sheet "Subjects", first column = subject, columns:
        Lignée, Acronyme lignée, Marque, Sexe, D. naissance,
        Chirurgie 1, D. chirurgie 1, Chirurgie 2, D. chirurgie 2,
        Virus, Dilution virus, Génotype, Espèce, Souche

Folder layout (built from the table, YYYY_MM_DD = session[:10]):

  raw session       <raw_root>/<dataset>/<YYYY_MM_DD>/<session>/
                        TSeries-*/ (with the Bruker .xml, exactly one)
                        <session>_rotary.abf
                        <session>_metadata.txt
  processed session <processed_root>/preprocessing/<YYYY_MM_DD>/<session>/
                        suite2p/
                        FaceIt/
  output            <processed_root>/NWBs/
"""
import os, sys, pathlib, glob, argparse
import numpy as np
import pandas as pd
import pynwb
import datetime
from dateutil.tz import tzlocal
import pyabf, json
from scipy.ndimage import gaussian_filter1d

from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.imaging.suite2p.to_nwb import add_ophys_processing_from_suite2p
from physion.utils.files import get_files_with_extension
from physion.analysis.tools import resample_signal

RECORDINGS_REQUIRED_COLUMNS = ['session', 'subject', 'protocol',
                               'experimenter', 'include']

def read_table(filename):

    dataset = pd.read_excel(filename,
                            sheet_name='Recordings')

    subjects = pd.read_excel(filename,
                            sheet_name='Subjects', index_col=0)

    missing = [c for c in RECORDINGS_REQUIRED_COLUMNS
               if c not in dataset.columns]
    if missing:
        raise ValueError('Sheet "Recordings" of "%s" is missing the column(s): %s'
                         % (filename, ', '.join(missing)))

    return dataset, subjects

def is_empty(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) \
            or (isinstance(value, str) and value.strip()=='')

def to_bool(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ['true', 'yes', 'y', '1', 'x', 'oui']:
            return True
        if v in ['false', 'no', 'n', '0', '', 'non']:
            return False
        raise ValueError('Cannot interpret include="%s" as a boolean' % value)
    if is_empty(value):
        return False
    return bool(value)

def build_session_paths(session, dataset, raw_root, processed_root):
    """
    returns a dict with the raw / processed folders and files of a session,
    raises FileNotFoundError if any required input is missing
    """
    day = session[:10]

    raw_session = os.path.join(raw_root, dataset, day, session)
    processed_session = os.path.join(processed_root, 'preprocessing',
                                     day, session)

    for folder, label in [(raw_session, 'raw session'),
                          (processed_session, 'processed session')]:
        if not os.path.isdir(folder):
            raise FileNotFoundError('%s folder not found: "%s"' % (label, folder))

    tseries = [f for f in glob.glob(os.path.join(raw_session, 'TSeries-*'))
               if os.path.isdir(f)]
    if len(tseries)!=1:
        raise FileNotFoundError(\
            'expected exactly one TSeries-* folder in "%s", found %i%s' % (\
                raw_session, len(tseries),
                (': '+', '.join(os.path.basename(t) for t in tseries)) if tseries else ''))
    tseries_folder = tseries[0]

    xml_files = get_files_with_extension(tseries_folder, extension='.xml')
    if len(xml_files)==0:
        raise FileNotFoundError('no Bruker .xml file in "%s"' % tseries_folder)

    paths = {'raw_session': raw_session,
             'processed_session': processed_session,
             'tseries_folder': tseries_folder,
             'xml_file': xml_files[0],
             'abf_file': os.path.join(raw_session, '%s_rotary.abf' % session),
             'metadata_file': os.path.join(raw_session, '%s_metadata.txt' % session),
             'suite2p_folder': os.path.join(processed_session, 'suite2p'),
             'faceit_folder': os.path.join(processed_session, 'FaceIt')}

    for key in ['abf_file', 'metadata_file']:
        if not os.path.isfile(paths[key]):
            raise FileNotFoundError('required file not found: "%s"' % paths[key])
    for key in ['suite2p_folder', 'faceit_folder']:
        if not os.path.isdir(paths[key]):
            raise FileNotFoundError('required folder not found: "%s"' % paths[key])

    return paths

def get_face_metrics(faceit_folder):

    fn_list = get_files_with_extension(faceit_folder, extension='.npz')
    fn_list = list(filter(lambda item: "faceit.npz" in item, fn_list))
    if len(fn_list)==0:
        raise FileNotFoundError('no "*faceit.npz" file in "%s"' % faceit_folder)
    fn = fn_list[0]
    faceitOutput = np.load(fn, allow_pickle=True)

    output = {}
    output['cx'] = faceitOutput['pupil_center_X']
    output['cy'] = faceitOutput['pupil_center_y']

    # WE TRICK THE PUPIL COORDINATES TO HAVE SOMETHING CORRECTED:
    output['sx'] = faceitOutput['pupil_dilation_blinking_corrected']/2./np.pi # faceitOutput['width']
    output['sy'] = np.ones(len(output['cx'])) # faceitOutput['height']
    output['blinking'] = faceitOutput['blinking_ids']

    output['grooming'] = faceitOutput['grooming_ids']
    output['face-motion'] = faceitOutput['motion_energy_without_grooming']

    return output

def get_running_speed(abf_file,
                      bin_conversion_th=1.5,
                      perimeter_cm=29,
                      cpr=1000,
                      new_freq=30,
                      position_smoothing=10e-3):

    abf = pyabf.ABF(abf_file)

    # Get acquisition frequency
    acq_freq = abf.sampleRate

    # Get channels
    trace_A, trace_B, t = get_abf_channels(abf, ['RE1', 'RE2'])

    # Concert to binary
    bin_A = np.where(trace_A >= bin_conversion_th, 1, 0)
    bin_B = np.where(trace_B >= bin_conversion_th, 1, 0)

    # Compute position
    position = compute_position_from_binary_signals(bin_A, bin_B, forward='counterclockwise')*perimeter_cm/cpr/4. #counts per revolution is cpr * 4
    if position_smoothing>0:
        position = gaussian_filter1d(position, int(position_smoothing*acq_freq), mode='nearest')

    # Compute position
    speed_vector = np.gradient(position, 1./acq_freq)
    speed = np.abs(speed_vector) #get the norm of the speed vector
    
    # Downsample the speed
    t, speed = resample_signal(speed, 
                               t_sample=t,
                               new_freq=new_freq)
    
    # Get start time of calcium imaging recording relative to rotary encoder recording
    abf.setSweep(sweepNumber=0, channel=0)
    epoch_idx = abf.sweepEpochs.types.index('Pulse')
    dt_start = abf.sweepEpochs.p1s[epoch_idx] / abf.sampleRate

    return t, speed, dt_start

def get_abf_channels(abf: pyabf.ABF, channels_names: list[str]=['RE1', 'RE2']):
    
    channel_a_idx = abf.adcNames.index(channels_names[0])
    channel_b_idx = abf.adcNames.index(channels_names[1])
    
    abf.setSweep(sweepNumber=0, channel=channel_a_idx)
    ch_A = abf.sweepY

    abf.setSweep(sweepNumber=0, channel=channel_b_idx)
    ch_B = abf.sweepY

    t = abf.sweepX

    return ch_A, ch_B, t

def compute_position_from_binary_signals(A, B, forward='counterclockwise'):
    '''
    Takes traces A and B and converts it to a trace that has the same number of
    points but with positions points.

    Algorithm based on the schematic of cases shown in the doc
    ---------------
    Input:
        A, B - traces to convert
   
    Output:
        Positions through time

    '''

    Delta_position = np.zeros(len(A)-1, dtype=float) # N-1 elements

    ################################
    ## positive_increment_cond #####
    ################################
    # The A signal lead the B signal (counterclockwise)
    # ... => 11 => 01 => 00 => 10 => 11 => ...
    PIC = ( (A[:-1]==1) & (B[:-1]==1) & (A[1:]==0) & (B[1:]==1) ) | \
        ( (A[:-1]==0) & (B[:-1]==1) & (A[1:]==0) & (B[1:]==0) ) | \
        ( (A[:-1]==0) & (B[:-1]==0) & (A[1:]==1) & (B[1:]==0) ) | \
        ( (A[:-1]==1) & (B[:-1]==0) & (A[1:]==1) & (B[1:]==1) )
    Delta_position[PIC] = 1

    ################################
    ## negative_increment_cond #####
    ################################
    # The B signal lead the A signal (clockwise)
    # ... => 11 => 10 => 00 => 01 => 11 => ...
    NIC = ( (A[:-1]==1) & (B[:-1]==1) & (A[1:]==1) & (B[1:]==0) ) | \
        ( (A[:-1]==1) & (B[:-1]==0) & (A[1:]==0) & (B[1:]==0) ) | \
        ( (A[:-1]==0) & (B[:-1]==0) & (A[1:]==0) & (B[1:]==1) ) | \
        ( (A[:-1]==0) & (B[:-1]==1) & (A[1:]==1) & (B[1:]==1) )
    Delta_position[NIC] = -1

    if forward=='clockwise':
        Delta_position = -Delta_position

    return np.cumsum(np.concatenate([[0], Delta_position]))

def format_virus(subject_info):
    virus = subject_info.get('Virus')
    dilution = subject_info.get('Dilution virus')
    if is_empty(virus):
        return None
    if is_empty(dilution):
        return str(virus)
    return '%s (dilution %s)' % (virus, dilution)

def convert(paths, session, subject,
            subject_info, experimenter,
            nwb_folder):

    Tseries_folder = paths['tseries_folder']

    # load metadata:
    with open(paths['metadata_file'], 'r') as f:
        metadata = json.load(f)

    metadata.update({'protocol': 'None'})

    # load Prairie file
    xml = bruker_xml_parser(paths['xml_file'])
    time = [int(t[:2]) for t in xml['StartTime'].split(':')]
    day = [int(d) for d in session[:10].split('_')]

    # session name YYYY_MM_DD-HH-MM-SS (zero-padded), as read by Post2p: NWBs/<session>.nwb
    nwb_filename = '%i_%.2d_%.2d-%.2d-%.2d-%.2d.nwb' % (*day, *time)

    start_time = datetime.datetime(day[0], day[1], day[2], 
                                   time[0], time[1], time[2], 
                                   tzinfo=tzlocal())

    # --------------------------------------------------------------
    #    ---------  building the pynwb subject object   ----------
    # --------------------------------------------------------------

    date_of_birth_list = [int(d) for d in subject_info['D. naissance'].split('_')]
    date_of_birth = datetime.datetime(date_of_birth_list[0], date_of_birth_list[1], date_of_birth_list[2], tzinfo=tzlocal())

    subject = pynwb.file.Subject(description=f"Mark: {subject_info['Marque']}, " \
                                 + f"Line: {subject_info['Lignée']} ({subject_info['Acronyme lignée']}), " \
                                 + f"Surgery 1: {subject_info['Chirurgie 1']} on {subject_info['D. chirurgie 1']}, " \
                                 + f"Surgery 2: {subject_info['Chirurgie 2']} on {subject_info['D. chirurgie 2']}",
                                    age= f"P{(start_time.date() - date_of_birth.date()).days}D",
                                    subject_id=subject,
                                    sex=subject_info['Sexe'],
                                    genotype=None if is_empty(subject_info.get('Génotype')) \
                                                else str(subject_info['Génotype']),
                                    date_of_birth=date_of_birth,
                                    species='mus musculus',
                                    strain=subject_info['Souche'])
    
    #raise ValueError('STOP - update the subject description with the correct metadata fields from the excel file (see code)')
    # --------------------------------------------------------------
    #    ---------  building the pynwb NWBfile object   ----------
    # --------------------------------------------------------------
    nwbfile = pynwb.NWBFile(\
                identifier=paths['raw_session'],
                session_description=str(metadata),
                experiment_description='imaging during spontaneous behavior',
                experimenter=experimenter,
                lab='Bacci lab',
                # protocol='spontaneous-activity',
                institution='Paris Brain Institute',
                virus=format_virus(subject_info),
                surgery='Viral-Injection+Headplate-Implantation',
                session_start_time=start_time,
                subject=subject,
                source_script=str(pathlib.Path(__file__).resolve()),
                source_script_file_name=str(pathlib.Path(__file__).resolve()),
                file_create_date=\
                   datetime.datetime.now(datetime.UTC).replace(tzinfo=tzlocal()))

    manager = pynwb.get_manager() # we need a manager to link raw and processed data

    # #################################################
    # ####         Locomotion                   #######
    # #################################################

    t_speed, speed, dt_start = get_running_speed(paths['abf_file'])

    running = pynwb.TimeSeries(name='Running-Speed',
                               data = np.reshape(speed, (len(speed),1)),
                               timestamps=t_speed,
                               unit='cm/s')
    nwbfile.add_acquisition(running)


    #################################################
    #### Pupil and Facemotion from FaceCamera #######
    #################################################

    t_imaging = xml['Green']['relativeTime'] + dt_start

    faceMetrics = get_face_metrics(paths['faceit_folder'])
    t_facedata = np.linspace(t_imaging[0], t_imaging[-1], len(faceMetrics['cx']))

    # Pupil
    pupil_module = nwbfile.create_processing_module(name='Pupil',
                                                    description='')
    
    for key in ['cx', 'cy', 'sx', 'sy', 'blinking']:
        PupilProp = pynwb.TimeSeries(name=key,
                    data = np.reshape(faceMetrics[key],
                                      (len(t_facedata),1)),
                    unit='seconds',
                    timestamps=t_facedata)
        pupil_module.add(PupilProp)

    # FaceMotion
    faceMotion_module = nwbfile.create_processing_module(name='FaceMotion', 
                                                         description='')
    
    for key in ['face-motion', 'grooming']:
        faceMotionProp = pynwb.TimeSeries(name=key,
                    data = np.reshape(faceMetrics[key],
                                    (len(t_facedata),1)),
                    unit='seconds',
                    timestamps=t_facedata)
        faceMotion_module.add(faceMotionProp)


    #################################################
    ####         Adding Imaging               #######
    #################################################
    functional_chan = 'Ch1'
    laser_key = 'Excitation 1'
    Depth = float(xml['settings']['positionCurrent']['ZAxis'])

    device = pynwb.ophys.Device(\
        'Imaging device with settings %s' %\
         str(xml['settings']).replace(': ','= '))
    nwbfile.add_device(device)
    optical_channel = pynwb.ophys.OpticalChannel(\
            'excitation_channel 1',
             laser_key,
             float(xml['settings']['laserWavelength'][laser_key]))

    imaging_plane = nwbfile.create_imaging_plane(\
            'my_imgpln', optical_channel,
                description='Depth=%.1f[um]' % Depth,
                device=device,
                excitation_lambda=float(xml['settings']['laserWavelength'][laser_key]),
                imaging_rate=1./float(xml['settings']['framePeriod']),
                indicator='GCamp',
                location='V1', # ADD METADATA HERE
                # reference_frame='A frame to refer to',
                grid_spacing=(\
                        float(xml['settings']['micronsPerPixel']['YAxis']),
                        float(xml['settings']['micronsPerPixel']['XAxis'])))

    image_series = pynwb.ophys.TwoPhotonSeries(\
           name='CaImaging-TimeSeries',
           dimension=[2], 
           data=np.ones((2,2,2)),
           imaging_plane=imaging_plane, 
           unit='s', 
           timestamps=1.*np.arange(2), # ADD UPDATE OF starting_time
           comments='raw-data-folder=%s' % Tseries_folder.replace('/', '**')) # TEMPORARY
    
    nwbfile.add_acquisition(image_series)

    print('=> Adding the suite2p processing from "%s" [...]' % paths['suite2p_folder'])
    add_ophys_processing_from_suite2p(paths['suite2p_folder'],
                                      nwbfile, xml,
                                      TwoP_trigger_delay=dt_start,
                                      device=device,
                                      optical_channel=optical_channel,
                                      imaging_plane=imaging_plane,
                                      image_series=image_series)

    #################################################
    ####         Writing NWB file             #######
    #################################################

    filename = os.path.join(nwb_folder, nwb_filename)
    io = pynwb.NWBHDF5IO(filename,
                         mode='w', manager=manager)

    print("""     ----> Saving the NWB file: "%s" """ % filename)
    io.write(nwbfile, link_data=False)
    io.close()
    print('---> done !')


if __name__=='__main__':

    parser = argparse.ArgumentParser(\
            description='Convert spontaneous two-photon recordings to NWB',
            epilog=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datatable',
                        help='DataTable.xlsx with "Recordings" and "Subjects" sheets')
    parser.add_argument('--raw-root', required=True,
                        help='root of the raw data, e.g. Y:\\raw\\two-photon')
    parser.add_argument('--processed-root', required=True,
                        help="the project's processed-data folder")
    parser.add_argument('--dataset', default=None,
                        help='dataset used for rows whose "dataset" cell is empty')
    args = parser.parse_args()

    dataset, subjects = read_table(args.datatable)

    nwb_folder = os.path.join(args.processed_root, 'NWBs')
    os.makedirs(nwb_folder, exist_ok=True)

    for irow, row in dataset.iterrows():

        session = str(row['session']).strip()
        label = 'row %i (session "%s")' % (irow+2, session) # +2: excel header + 1-based

        if not to_bool(row['include']):
            print('\n-- skipping %s: include is False' % label)
            continue

        protocol = '' if is_empty(row['protocol']) else str(row['protocol']).strip()
        if protocol.lower()!='spontaneous':
            print('\n-- skipping %s: protocol is "%s", not "spontaneous"' % (label, protocol))
            continue

        row_dataset = row['dataset'] if 'dataset' in dataset.columns else None
        if is_empty(row_dataset):
            row_dataset = args.dataset
        if is_empty(row_dataset):
            raise ValueError('%s: empty "dataset" cell and no --dataset given' % label)

        if is_empty(row['experimenter']):
            raise ValueError('%s: empty "experimenter" cell' % label)

        subject = row['subject']
        if subject not in subjects.index:
            raise KeyError('%s: subject "%s" not found in the "Subjects" sheet'
                           % (label, subject))
        subject_info = subjects.loc[subject].to_dict()

        paths = build_session_paths(session, str(row_dataset).strip(),
                                    args.raw_root, args.processed_root)

        print("\n" + paths['raw_session'])
        convert(paths, session, str(subject), subject_info,
                str(row['experimenter']).strip(), nwb_folder)

