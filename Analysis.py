import matplotlib.pyplot as plt
import numpy
import pandas

from Simulations import load_obj
from Simulations import load_reporters


def plot_ampvul():
    reporters = load_reporters()
    vul_list = load_obj('vul_list')
    deletion_results = load_obj('node_deletion')

    db = pandas.concat(deletion_results)
    # Dropping because recorded each item in list rather than sum
    db = db.drop(columns=['final_agg_loss']).drop_duplicates()
    commodities = pandas.unique(db.commodity.values.ravel())
    commodities.sort()

    reporters_to_i = {key: value for (value, key) in enumerate(reporters)}
    # print(len(commodities))
    # print(commodities)

    # Need to query amp for comm
    for year, comm_vul in vul_list:
        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        relevant_rows = (db.loc[db['year'] == year])
        # print(amplifications)
        for comm_i, vuls in enumerate(comm_vul):
            comm = commodities[comm_i]
            plt.title('Amplification vs Vulnerability Comm: ' + str(comm))
            plt.xlabel('Vulnerability')
            plt.ylabel('Amplification')

            amplifications_df = \
            (relevant_rows.loc[(relevant_rows['commodity'] == comm) & (relevant_rows['comm_amplication'] >= 0)])[
                ['country', 'comm_amplication']]
            amplifications = [tuple(x) for x in amplifications_df.values]
            reps_in_amp = [reporters_to_i[x[0]] for x in amplifications]
            amplifications = [x for _, x in amplifications]
            amp_tmp = numpy.array(amplifications)
            amp_mean = numpy.mean(amp_tmp, axis=0)
            amp_sd = numpy.std(amp_tmp, axis=0)

            # print(vuls[0].tolist())
            tmp_vul = [v for i, v in enumerate(vuls[0].tolist()) if i in reps_in_amp]
            vul_tmp = numpy.array(tmp_vul)
            vul_mean = numpy.mean(vul_tmp)
            vul_sd = numpy.std(vul_tmp)

            # print(len(amplifications))
            # print(len(tmp_vul))

            combined = [(x, y * 10) for x, y in zip(amplifications, tmp_vul) if
                        outlier_check(amp_mean, amp_sd, x, 2) & outlier_check(vul_mean, vul_sd, y, 2)]

            new_amp = [x for x, _ in combined]
            new_vul = [y for _, y in combined]
            # print('new', len(new_amp), len(new_vul))

            plt.scatter(new_vul, new_amp)

            filename = 'results/' + str(year) + '_' + str(comm) + '_amp_vul.png'
            plt.savefig(filename)
            print('Saving to file', filename)
            plt.clf()

        # print(len(amplifications), len(comm_vul))
        # plt.scatter(amplifications, comm_vul, label=str(comm))

        # filename = 'results/' + str(year) + '_amp_vul.png'
        # plt.legend(loc='best')
        # plt.savefig(filename)
        # print('Saving to file', filename)
        # plt.clf()


def plot_avampvul():
    reporters = load_reporters()
    vul_list = load_obj('vul_list')
    deletion_results = load_obj('node_deletion')

    db = pandas.concat(deletion_results)
    # Dropping because recorded each item in list rather than sum
    db = db.drop(columns=['final_agg_loss']).drop_duplicates()
    commodities = pandas.unique(db.commodity.values.ravel())
    commodities.sort()

    for year, comm_vuls in vul_list:
        relevant_rows = (db.loc[db['year'] == year])

        av_vuls = numpy.zeros((len(comm_vuls[0]))).tolist()

        for comm_i, vuls in enumerate(comm_vuls):
            # print(comm_i)
            for i, v in enumerate(vuls[0].tolist()):
                if v > 0:
                    av_vuls[i] += v

        av_vuls = [x / len(comm_vuls) for x in av_vuls]

        av_amps = []

        for reporter in reporters:
            amplifications_df = \
                (relevant_rows.loc[(relevant_rows['country'] == reporter) & (relevant_rows['comm_amplication'] >= 0)])[
                    ['comm_amplication']]
            av_amps.append((amplifications_df.mean())[0])

        vul_tmp = numpy.array(av_vuls)
        vul_mean = numpy.mean(vul_tmp)
        vul_sd = numpy.std(vul_tmp)

        amp_tmp = numpy.array(av_amps)
        amp_mean = numpy.mean(amp_tmp)
        amp_sd = numpy.std(amp_tmp)

        plt.title('Average Amplification vs Vulnerability')
        plt.xlabel('Vulnerability (x10 for scaling)')
        plt.ylabel('Amplification')

        combined = [(x, y) for x, y in zip(av_amps, av_vuls) if
                    outlier_check(amp_mean, amp_sd, x, 1.5) & outlier_check(vul_mean, vul_sd, y, 1.5)]

        new_amp = [x for x, _ in combined]
        new_vul = [y for _, y in combined]

        plt.scatter(new_vul, new_amp)

        filename = 'results/' + str(year) + '_av_amp_vul.png'
        plt.savefig(filename)
        print('Saving to file', filename)
        plt.clf()


def outlier_check(mean, sd, val, sds):
    if (val > mean + sds * sd) | (val < mean - sds * sd):
        print("Removing", val)
        return False
    else:
        return True


# plot_ampvul()
plot_avampvul()
