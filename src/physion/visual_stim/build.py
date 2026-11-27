import sys
import cv2 as cv
import numpy as np

import physion

def build_stim(protocol, 
               from_file=None):
    """
    """
    if from_file is not None:
        # we only build the time course of the associated video
        return physion.visual_stim.main.visual_stim(\
                                                protocol, 
                                        from_file=from_file)

    elif (protocol['Presentation']=='multiprotocol'):
        #
        return physion.visual_stim.main.multiprotocol(\
                                                protocol)

    else:
        # single protocol
        protocol_name = protocol['Stimulus'].replace('-', '_').replace('+', '_')
        try:
            return getattr(\
                        getattr(\
                            physion.visual_stim.stimuli,\
                                            protocol_name),
                                              'stim')(protocol)
        except ModuleNotFoundError:
            print('\n [!!] Protocol not recognized ! [!!] \n ')
            return None

def get_default_params(protocol_name):
    """
    """
    protocol_name = protocol_name.replace('-',
                                    '_').replace('+', '_')

    try:
        Dparams = getattr(\
                        getattr(\
                            physion.visual_stim.stimuli, protocol_name), 
                                'params')
        params = {}
        # set all params to default values
        for key in Dparams:
            params[key] = Dparams[key]

        params['Stimulus'],  params['Presentation'] = protocol_name, ''
        params['Screen'], params['demo'] = 'Dell-2020', True
        params['seed-1'], params['seed-2'] = 1, 1
        params['N-seed'] = 1
        params['N-repeat'] = 1

        if 'presentation-duration' not in params:
            params['presentation-duration'] = 3

        params['presentation-blank-screen-color'] = 0.5

        params['presentation-prestim-period'] = 0.5
        params['presentation-interstim-period'] = 0.5
        params['presentation-poststim-period'] = 0.5

        params['movie_refresh_freq'] = 30.
        params['units'] = 'cm'

        return params

    except ModuleNotFoundError:

        print('\n [!!] Protocol not recognized ! [!!] \n ')
        return None

class MonitoringSquare:
    """
    Used to draw a flickering square on a corner of the screen
    to monitor stimulus presentation during the experiment
    """

    def __init__(self, Stim):

        self.find_mask(Stim)

    def draw(self, image, t, tstart, tstop):
        """
        't' is the time from stimulus start !
        """
        image[self.mask] = -1. # black by default
        if (t<tstop) & (t>=tstart):
            iT = int(1000*(t-tstart))
            if (iT%1000)<150:
                image[self.mask] = 1.
            elif (iT>=500) & (iT<650):
                image[self.mask] = 1.
        return image

    def find_mask(self, Stim):
        """ find the position of the square """

        self.mask = np.array(Stim.get_null_image(),
                             dtype=bool)

        S = int(Stim.screen['monitoring_square']['size'])
        X, Y = Stim.screen['resolution'] # x,y sizes

        if Stim.screen['monitoring_square']['location']=='top-right':
            self.mask[X-S:,Y-S:] = True
        elif Stim.screen['monitoring_square']['location']=='top-left':
            self.mask[X-S:,:S] = True
        elif Stim.screen['monitoring_square']['location']=='bottom-right':
            self.mask[X-S:,:S] = True
        elif Stim.screen['monitoring_square']['location']=='bottom-left':
            self.mask[:S,:S] = True
        else:
            print(30*'-'+'\n [!!]  monitoring square location not recognized !!')
            print('        --> (0,0) by default \n')
            self.mask[:S,:S] = True
        


#######################################################
##  ----  saving stimulus images (--save-images)  ---- ##
#######################################################

# experiment keys that are not stimulus parameters
NON_STIM_KEYS = ['index', 'repeat', 'protocol_id', 'bg-color',
                 'time_start', 'time_stop', 'time_duration',
                 'interstim', 'interstim-screen']

def get_varied_keys(Stim):
    """ stimulus parameters that vary across episodes """
    keys = []
    for k in Stim.experiment:
        if k in NON_STIM_KEYS:
            continue
        vals = [v for v in Stim.experiment[k] if v is not None]
        try:
            if len(np.unique(np.array(vals, dtype=float)))>1:
                keys.append(k)
        except (TypeError, ValueError):
            pass # non-numeric parameter
    return keys

def stim_key(Stim, episode):
    """ identifies a distinct stimulus (repeats share the same key) """
    return (int(Stim.experiment['protocol_id'][episode]),
            int(Stim.experiment['index'][episode]))

def stim_image_name(Stim, episode, keys):
    """ e.g. "stim-006_Image-ID-6_contrast-0.3" """
    pid, index = stim_key(Stim, episode)
    name = 'stim-%03d' % index
    if Stim.protocol['Presentation']=='multiprotocol':
        name = 'protocol-%i_' % (pid+1) + name
    for k in keys:
        v = Stim.experiment[k][episode]
        if v is not None:
            name += '_%s-%g' % (k, float(v))
    return name


if __name__=='__main__':

    import argparse, os, pathlib, shutil, json

    parser=argparse.ArgumentParser()
    parser.add_argument("protocol", 
                        help="protocol as a json file", 
                        default='')
    parser.add_argument("--mp4", 
                        help="force to mp4 instead of wmv", 
                        action="store_true")
    parser.add_argument('-v', "--verbose",
                        action="store_true")
    parser.add_argument("--save-images",
                        help="""save the first frame of each distinct stimulus
                        as a png (exactly as written in the movie)
                        in the "stim-images" subfolder""",
                        action="store_true")
    args = parser.parse_args()

    if os.path.isfile(args.protocol) and args.protocol.endswith('.json'):
            
            # create the associated protocol folder in the movies folder
            protocol_folder = \
                os.path.join(os.path.dirname(args.protocol),
                    'movies',
                    os.path.basename(args.protocol.replace('.json','')))

            if os.path.isfile(os.path.join(protocol_folder, 'protocol.json')):
                # remove the previous content for security
                shutil.rmtree(protocol_folder)

            # re-create an empty one
            pathlib.Path(protocol_folder).mkdir(\
                                    parents=True, exist_ok=True)

            # build the protocol
            with open(args.protocol, 'r') as f:
                protocol = json.load(f)

            protocol['json_location'] = os.path.dirname(args.protocol)

            if args.verbose:
                protocol['verbose'] = True

            Stim = build_stim(protocol)

            #  copy the protocol infos
            with open(os.path.join(protocol_folder, 'protocol.json'), 'w') as f:
                json.dump(Stim.protocol, f, indent=4)

            def update(Stim, index):
                if index<len(Stim.experiment['index']):
                    print(' - episode %i/%i ' % (\
                            index+1, len(Stim.experiment['index'])),
                          '   duration: ', 
                        #   Stim.experiment['time_duration'][index],
                        #   '   contrast: ', 
                        #   Stim.experiment['contrast'][index],
                        #   '   protocol-id : ', 
                          Stim.experiment['protocol_id'][index])
                    if 'verbose' in protocol:
                        for k in Stim.experiment:
                            print(18*' '+'- %s:%.1f ' %\
                                    (k, Stim.experiment[k][index]))
                    tstart = Stim.experiment['time_start'][index]
                    tstop= Stim.experiment['time_stop'][index]
                    return tstart, tstop
                else:
                    return np.inf, np.inf

            # prepare the monitoring square properties
            if 'monitoring_square' in Stim.screen:
                square = MonitoringSquare(Stim)

            # prepare video file
            Format = 'mp4' if (('linux' in sys.platform) or args.mp4) else 'wmv'
            out = []
            for s in range(Stim.screen['nScreens']):

                if s==0 and Stim.screen['nScreens']==1:
                    filename = os.path.join(protocol_folder, 
                                            'movie.%s' % Format)
                else:
                    filename = os.path.join(protocol_folder, 
                                            'movie-%i.%s' % (s+1, Format))
                print("""
                    Screen %i, %s""" % (s+1, filename))
                
                out.append(cv.VideoWriter(filename,
                                cv.VideoWriter_fourcc(*'mp4v'), 
                                Stim.movie_refresh_freq,
                                Stim.screen['resolution'],
                                False))
            print()
            # prepare the loop
            t, tend = 0, Stim.experiment['time_stop'][-1]+\
                    Stim.experiment['interstim'][-1]
            index = 0
            tstart, tstop = update(Stim, index)

            # prepare the saving of stimulus images
            if args.save_images:
                images_folder = os.path.join(protocol_folder, 'stim-images')
                pathlib.Path(images_folder).mkdir(parents=True, exist_ok=True)
                img_keys = get_varied_keys(Stim)
                saved_stims = set()

            # LOOP over time to build the movie
            while t<tend:

                if t>=tstop:
                    index += 1
                    tstart, tstop = update(Stim, index)
                    
                # data in [0,1]
                if (t>=tstart) and (t<tstop):
                    data = Stim.gamma_correction(\
                            Stim.get_image(index, t-tstart))
                else:
                    data = Stim.blank_color+\
                                Stim.get_null_image()

                # put the monitoring square
                if 'monitoring_square' in Stim.screen:
                    data = square.draw(data, t, tstart, tstop)

                for s, o in enumerate(out):
                    dataS = Stim.restrict_to_screen(data, 
                                                    screen_id=s+1)
                    frame = np.array(255*np.rot90(dataS, k=1),
                                     dtype='uint8')
                    o.write(frame)

                    if args.save_images and (t>=tstart) and (t<tstop) and\
                            (stim_key(Stim, index) not in saved_stims):
                        name = stim_image_name(Stim, index, img_keys)
                        if len(out)>1:
                            name += '_screen-%i' % (s+1)
                        cv.imwrite(os.path.join(images_folder, name+'.png'),
                                   frame)

                if args.save_images and (t>=tstart) and (t<tstop):
                    saved_stims.add(stim_key(Stim, index))

                t+= 1./Stim.movie_refresh_freq

            for o in out:
                o.release()

            np.save(os.path.join(protocol_folder, 'visual-stim.npy'), 
                    Stim.experiment)
            print('\n [ok] video file and metadata saved in: "%s" \n ' % protocol_folder)

    else:
        print('\nERROR: need to provide a valid json file as argument\n')
