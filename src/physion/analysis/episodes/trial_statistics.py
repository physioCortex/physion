# general modules
import os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import pearsonr, sem
import random
from sklearn import metrics

# custom modules
import physion.utils.plot_tools as pt
from physion.analysis import stat_tools

def stat_test_for_evoked_responses(ep,
                                   episode_cond=None,
                                   response_args={},
                                   interval_pre=[-2,0], 
                                   interval_post=[1,3],
                                   test='wilcoxon',
                                   sign='positive',
                                   verbose=True):
        """
        Takes EpisodeData
        Choose quantity from where you want to do a statistical test
            through "response_args"
            check possibilities with ep.quantities

        Choose the test you want . default wilcoxon . 

        It performs a test between the values from \
                interval_pre and interval_post
        
        returns a StatTest object see../stat_test.py
        """

        response = ep.get_response2D(episode_cond = episode_cond,
                                     **response_args)

        pre_cond  = ep.compute_interval_cond(interval_pre)
        post_cond  = ep.compute_interval_cond(interval_post)

        if len(response.shape)>1:
            return stat_tools.StatTest(response[:,pre_cond].mean(axis=1),
                                       response[:,post_cond].mean(axis=1),
                                       test=test, 
                                       sign=sign,
                                       verbose=verbose)
        else:
            return None


def run_pre_post_stat(ep,
        merged_episode_cond,
        response_args,
        response_significance_threshold=0.05,
        stat_test_props={},
        verbose=True):
    """
    Docstring for run_pre_post_stat

    reformatting of stat_test_for_evoked_responses
    so that it returns a dictionary
    """

    stats = ep.stat_test_for_evoked_responses(\
                            episode_cond=merged_episode_cond,
                                    response_args=response_args,
                                            verbose=verbose,
                                                **stat_test_props)
    
    if stats is not None:
        return {
            'value' : np.mean(stats.y-stats.x),
            'std-value' : np.std(stats.y-stats.x),
            'ntrials' : len(stats.x),
            'pval' : stats.pvalue,
            'significant': stats.significant(\
                                threshold=response_significance_threshold),
        }

    else:
        return {}


def build_episode_params_variations(ep, repetition_keys):
    """
    Docstring for build_episode_params_variations

    find the keys and values of the different stimulus
    parameters within a 

    exclude the "repetition_keys" in splitting the episodes
    e.g.    
        - repetition_keys=['repeat'] (default)
        so that the set of different episodes per condition
        is just made of the stimuli with different "repeat" values
        - repetition_keys=['repeat','angle']
        the set of different episodes per condition
        is now made of the stimuli with different "repeat" and
            "angle" (i.e. stim. orientation) values
    
    """

    VARIED_KEYS, VARIED_VALUES = [], []
    for key in ep.varied_parameters:
        if key not in repetition_keys:
            VARIED_KEYS.append(key)
            VARIED_VALUES.append(ep.varied_parameters[key])

    return VARIED_KEYS, VARIED_VALUES


def calc_pval_factor(VARIED_VALUES, 
                     multiple_comparison_correction):
    """
    Docstring for calc_pval_factor

    when multiple_comparison_correction==True
        --> Bonferroni correction, 
            i.e. just divide by number of comparisons
    
    """

    nStims = len(np.meshgrid(*VARIED_VALUES)[0].flatten())

    if multiple_comparison_correction:

        return 1./nStims, nStims

    else:

        return 1., sStims

def run_analysis_splitting_by_stim_params(ep, 
                                          stat_func,
                                          response_args,  
                                          episode_cond=None,
                                          stat_test_props={},
                                          repetition_keys=['repeat'],
                                          multiple_comparison_correction=True,
                                          response_significance_threshold=0.05,
                                          nMin_episodes=5,
                                          verbose=True):
    """
    Docstring for run_analysis_splitting_by_stim_params
    
    :param ep: Description
    :param stat_func: Description
    :param full_summary: Description
    :param response_args: Description
    :param episode_cond: Description
    :param stat_test_props: Description
    :param repetition_keys: Description
    :param nMin_episodes: Description
    :param verbose: Description
    """

    VARIED_KEYS, VARIED_VALUES = \
        build_episode_params_variations(ep, 
                                        repetition_keys)

    full_summary = {}

    for key in VARIED_KEYS+\
            ['value', 'std-value', 'ntrials', 'pval', 'significant',
             'statistic', 'r']:
        full_summary[key] = []

    if len(VARIED_KEYS)>0:

        pval_factor, nStims = \
              calc_pval_factor(VARIED_VALUES,
                               multiple_comparison_correction)

        if verbose:
            print(' %i different stimulus configurations, ' % nStims)
            print('    running stat analysis one by one [...]')
            print()

        for values in itertools.product(*VARIED_VALUES):

            merged_episode_cond = episode_cond &\
                        ep.find_episode_cond(key=VARIED_KEYS, 
                                             value=values)

            if np.sum(merged_episode_cond)>=nMin_episodes:

                # store stim params values
                for key, value in zip(VARIED_KEYS, values):
                    full_summary[key].append(value)

                # perform stat analysis, from "stat_func" argument
                summary = stat_func(ep,
                    merged_episode_cond,
                    response_args,
                    response_significance_threshold=pval_factor*\
                                    response_significance_threshold,
                    stat_test_props=stat_test_props,
                    verbose=verbose)

                # store statistics values
                for key in summary:
                    full_summary[key].append(\
                        summary[key]
                        )

            else:

                if verbose:
                    print(' Number of episodes n=%i for cond:' % np.sum(merged_episode_cond))
                    print('      ', VARIED_KEYS, ' = ', values)
                    print('     is lower that nMin_episodes specific for stat_test (%i)' % nMin_episodes)
                    print('        ---> NOT INCLUDED int summary statistics')

    else:

        if np.sum(episode_cond)>=nMin_episodes:

            # perform stat analysis, from "stat_func" argument
            full_summary = stat_func(ep,
                episode_cond,
                response_args,
                response_significance_threshold=response_significance_threshold,
                stat_test_props=stat_test_props,
                verbose=verbose)

    for key in full_summary:
        full_summary[key] = np.array(full_summary[key])

    return full_summary

def pre_post_statistics(ep, stat_test_props,
                        episode_cond=None,
                        repetition_keys=['repeat'],
                        nMin_episodes=5,
                        response_args={},
                        response_significance_threshold=0.05,
                        multiple_comparison_correction=True,
                        loop_over_cells=False,
                        verbose=True):
    '''
    return all the statistic values organized in a dictionary (str keys and arrays of values). 
    dictionnary keys : 
            parameter keys +
            'value', 'std-value',  'significant', 'pval', 'ntrials', 

    when the minimum number of episodes is not met, 
        --> the condition is not included in the summary statistics

    by default, 
        Bonferroni correction for multiple comparison in significance
    '''

    if episode_cond is None:
        episode_cond = ep.find_episode_cond() # all true by default

    if not loop_over_cells:

        return run_analysis_splitting_by_stim_params(ep, 
                                            run_pre_post_stat,
                                            response_args,  
                                            episode_cond=episode_cond,
                                            stat_test_props=stat_test_props,
                                            repetition_keys=repetition_keys,
                                            multiple_comparison_correction=\
                                                multiple_comparison_correction,
                                            response_significance_threshold=\
                                                response_significance_threshold,
                                            nMin_episodes=nMin_episodes,
                                            verbose=verbose)

    else:

        summaries = []
        quantity = response_args['quantity']

        for cell in np.arange(getattr(ep, quantity).shape[1]):
            response_args = {'quantity':quantity,
                             'roiIndex':cell}
            summaries.append(\
                run_analysis_splitting_by_stim_params(ep, 
                                            run_pre_post_stat,
                                            response_args,  
                                            episode_cond=episode_cond,
                                            stat_test_props=stat_test_props,
                                            repetition_keys=repetition_keys,
                                            multiple_comparison_correction=\
                                                multiple_comparison_correction,
                                            response_significance_threshold=\
                                                response_significance_threshold,
                                            nMin_episodes=nMin_episodes,
                                            verbose=verbose)
            )

        # brings this back to arrays of shape (ROIs, ...)
        summary = {}

        for key in summaries[0]:

            if key in ['value', 'std-value', 'pval',
                        'significant', 'statistic', 'r']:
                summary[key] = np.array([s[key] for s in summaries])
            else:
                summary[key] = np.array(summaries[0][key])


        return summary


def run_reliability_test(ep,
                     merged_episode_cond,
                     response_args,
                     response_significance_threshold=0.05,
                     stat_test_props=dict(seed=1, n_samples=500),
                     return_samples=False,
                     verbose=True):
    """

    Compute the reliability using the method from T.D. Marks (2021) and C.G. Sweeney (2025). 
    To compute reliability, the function splits the trials randomly in two halves, trial-averages the two groups and calculates the Pearson's correlation. 
    The process is done n_samples times and averaged to get the reliability measure.
    """

    response = ep.get_response2D(episode_cond=merged_episode_cond,
                                 **response_args)

    # fix seed for reproducibility:
    np.random.seed(stat_test_props['seed'])

    corr_list = []
    null_corr_list = []
    if return_samples:
        real, shuffled = [], []

    set_trials = list(range(response.shape[0]))
    split = len(set_trials) // 2

    for _ in range(stat_test_props['n_samples']):

        # Divide randomly the trials in 2 groups
        random.shuffle(set_trials)
        group1 = set_trials[:split]
        group2 = set_trials[split:]
        
        # Shuffle circularly
        time_shifts = np.random.choice(np.arange(0, response.shape[1]), 
                                       len(response), replace=True)
        shifted_traces = np.array([np.roll(response[j], dt)\
                                    for j, dt in enumerate(time_shifts)])

        averaged_group1 = np.mean(response[group1, :], axis=0)
        averaged_group2 = np.mean(response[group2, :], axis=0)

        averaged_group1_null = np.mean(shifted_traces[group1, :], axis=0)
        averaged_group2_null = np.mean(shifted_traces[group2, :], axis=0)

        if return_samples:
            real.append((averaged_group1+averaged_group2)/2.)
            shuffled.append(.5*(averaged_group1_null+averaged_group2_null))

        corr = pearsonr(averaged_group1, averaged_group2)[0]
        corr_list.append(corr)

        corr_null = pearsonr(averaged_group1_null, averaged_group2_null)[0]
        null_corr_list.append(corr_null)

    percentile = 100-100.*response_significance_threshold
    perc_threshold = np.percentile(null_corr_list, percentile)

    # store summary data:
    summary = {}
    summary['r'] = np.mean(corr_list)
    summary['significant'] = summary['r'] > perc_threshold
    summary['pval'] = np.sum(np.array(null_corr_list) >= summary['r']) / len(null_corr_list)

    if return_samples:
        summary['corr_list'] = corr_list
        summary['null_corr_list'] = null_corr_list
        summary['real'] = real
        summary['shuffled'] = shuffled

    return summary


def reliability(ep,
                episode_cond=None,
                response_args={},
                repetition_keys=['repeat'],
                nMin_episodes=5,
                response_significance_threshold=0.05,
                multiple_comparison_correction=True,
                stat_test_props=dict(seed=1, n_samples=500),
                loop_over_cells=False,
                verbose=True):

    if episode_cond is None:
        episode_cond = ep.find_episode_cond() # all true by default

    if not loop_over_cells:

        return run_analysis_splitting_by_stim_params(ep, 
                                            run_reliability_test,
                                            response_args,  
                                            episode_cond=episode_cond,
                                            stat_test_props=stat_test_props,
                                            repetition_keys=repetition_keys,
                                            multiple_comparison_correction=\
                                                multiple_comparison_correction,
                                            response_significance_threshold=\
                                                response_significance_threshold,
                                            nMin_episodes=nMin_episodes,
                                            verbose=verbose)

    else:

        summaries = []
        quantity = response_args['quantity']

        for cell in np.arange(getattr(ep, quantity).shape[1]):
            response_args = {'quantity':quantity,
                             'roiIndex':cell}
            summaries.append(\
                run_analysis_splitting_by_stim_params(ep, 
                                            run_reliability_test,
                                            response_args,  
                                            episode_cond=episode_cond,
                                            stat_test_props=stat_test_props,
                                            repetition_keys=repetition_keys,
                                            multiple_comparison_correction=\
                                                multiple_comparison_correction,
                                            response_significance_threshold=\
                                                response_significance_threshold,
                                            nMin_episodes=nMin_episodes,
                                            verbose=verbose)
            )

        # brings this back to arrays of shape (ROIs, ...)
        summary = {}

        for key in summaries[0]:

            if key in ['value', 'std-value', 'ntrials', 'pval', 'significant', 'r']:
                summary[key] = np.array([s[key] for s in summaries])
            else:
                summary[key] = summaries[0][key]


        return summary

def find_bounds(index:int, bool_array:list):
    """
    Find the 'True' values interval boundaries surrounding the given index value.
    
    :param int index: Index to be considered.
    :param list bool_array: List of bool.
    
    :return start_idx (int): Starting index of the interval.
    :return end_idx (int): Ending index of the interval.
    """
    n = len(bool_array)
    
    i = index - 1
    while i >= 0 and bool_array[i] == True :
        i -= 1
    start_idx = i + 1

    i = index + 1
    while i < n and bool_array[i] == True :
        i += 1
    end_idx = i - 1

    return start_idx, end_idx
    
def visualpipe_evoked_responses(ep,
                episode_cond=None,
                response_args={},
                repetition_keys=['repeat'],
                nMin_episodes=7,
                sign='positive',
                auc_thr=5.,
                dt_response_min=0.2,
                response_significance_threshold=0.05,
                multiple_comparison_correction=True,
                stat_test_props=dict(seed=1, n_samples=500),
                loop_over_cells=False,
                verbose=True):
    
    if episode_cond is None:
        episode_cond = ep.find_episode_cond() # all true by default

    quantity = response_args['quantity']
    nb_frames_min = dt_response_min * ep.dt_sampling
    baselineCond = (ep.t<0)
    
    if not loop_over_cells:

        avg_response = getattr(ep, quantity).mean(axis=(0, 1))
        zscored_avg_response = ( (avg_response.T - avg_response[baselineCond].mean().T) / avg_response[baselineCond].std()).T

        max_val, min_val = np.max(zscored_avg_response[~baselineCond]), np.min(zscored_avg_response[~baselineCond])
        if max_val > 1 and max_val > np.abs(min_val):
            sign_roi = 'positive'
            start_idx, end_idx = find_bounds(np.argmax(zscored_avg_response[~baselineCond]), zscored_avg_response[~baselineCond] >= 0)
        elif min_val < -1 :
            sign_roi = 'negative'
            start_idx, end_idx = find_bounds(np.argmin(zscored_avg_response[~baselineCond]), zscored_avg_response[~baselineCond] <= 0)
        else :
            sign_roi = "non-responsive"
            start_idx, end_idx = 0, 0
        
        summary = run_analysis_splitting_by_stim_params(ep, 
                                            run_reliability_test,
                                            response_args,  
                                            episode_cond=episode_cond,
                                            stat_test_props=stat_test_props,
                                            repetition_keys=repetition_keys,
                                            multiple_comparison_correction=\
                                                multiple_comparison_correction,
                                            response_significance_threshold=\
                                                response_significance_threshold,
                                            nMin_episodes=nMin_episodes,
                                            verbose=verbose)
    
        #time duration constraint & AUC constraint
        auc = metrics.auc(ep.t[~baselineCond][start_idx:end_idx+1], zscored_avg_response[~baselineCond][start_idx:end_idx+1])
        if end_idx - start_idx + 1 < nb_frames_min or np.abs(auc) < auc_thr :
            summary['significant'] = False

        elif (sign!='both' and sign_roi != sign) or (sign=='both' and sign_roi == "non-responsive"):
            summary['significant'] = False

    else:

        summaries = []

        averaged_response = getattr(ep, quantity).mean(axis=0)
        zscored_avg_response = ( (averaged_response.T - averaged_response[:, baselineCond].mean(axis=1).T) / averaged_response[:, baselineCond].std(axis=1)).T

        for cell in np.arange(getattr(ep, quantity).shape[1]):

            max_val, min_val = np.max(zscored_avg_response[cell, ~baselineCond]), np.min(zscored_avg_response[cell, ~baselineCond])

            if max_val > 1 and max_val > np.abs(min_val):
                sign_roi = 'positive'
                start_idx, end_idx = find_bounds(np.argmax(zscored_avg_response[cell, ~baselineCond]), zscored_avg_response[cell, ~baselineCond] >= 0)
            elif min_val < -1 :
                sign_roi = 'negative'
                start_idx, end_idx = find_bounds(np.argmin(zscored_avg_response[cell, ~baselineCond]), zscored_avg_response[cell, ~baselineCond] <= 0)
            else :
                sign_roi = "non-responsive"
                start_idx, end_idx = 0, 0

            if (sign!='both' and sign_roi != sign) or (sign=='both' and sign_roi == "non-responsive"):
                summaries.append({'r': np.nan, 'significant': False, 'pval': np.nan})
                continue
            
            #time duration constraint & AUC constraint
            auc = metrics.auc(ep.t[~baselineCond][start_idx:end_idx+1], zscored_avg_response[cell, ~baselineCond][start_idx:end_idx+1])
            if end_idx - start_idx + 1 < nb_frames_min or np.abs(auc) < auc_thr :
                summaries.append({'r': np.nan, 'significant': False, 'pval': np.nan})
                continue

            response_args = {'quantity':quantity,
                             'roiIndex':cell}
            
            summaries.append(\
                run_analysis_splitting_by_stim_params(ep, 
                                            run_reliability_test,
                                            response_args,  
                                            episode_cond=episode_cond,
                                            stat_test_props=stat_test_props,
                                            repetition_keys=repetition_keys,
                                            multiple_comparison_correction=\
                                                multiple_comparison_correction,
                                            response_significance_threshold=\
                                                response_significance_threshold,
                                            nMin_episodes=nMin_episodes,
                                            verbose=verbose)
            )

        # brings this back to arrays of shape (ROIs, ...)
        summary = {}

        for key in summaries[0]:

            if key in ['value', 'std-value', 'ntrials', 'pval', 'significant', 'r']:
                summary[key] = np.array([s[key] for s in summaries])
            else:
                summary[key] = summaries[0][key]

    return summary


from .build import EpisodeData
EpisodeData.stat_test_for_evoked_responses = stat_test_for_evoked_responses
EpisodeData.pre_post_statistics = pre_post_statistics
EpisodeData.reliability = reliability
EpisodeData.visualpipe_evoked_responses = visualpipe_evoked_responses