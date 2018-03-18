import copy
import glob
import json
import os

import matplotlib.pyplot as pyplot
import networkx
import numpy
import pandas
import pyximport; pyximport.install()
from pandas.errors import EmptyDataError


data_dir = 'output/'
countries_file = 'resources/traders.json'  # File with trader information
partners_file = 'resources/partners.json'

IMPORT = 1
EXPORT = 2
REEXPORT = 3
REIMPORT = 4

YEARS = list(range(2000, 2001))


def run():
    # traders = loadcountries()
    partners = load_partners()
    db = setupdb()

    node_deletion_results = []
    for year in YEARS:
        imports, aggregate, reporters, comm_list = makematrices(db, partners, year)
        # print(imports)
        # print(exports)
        agg = matrix_calcs(aggregate)

        node_deletion_results.extend(node_deletion_sim(year, agg, imports, reporters, comm_list))

    node_deleteion_db = pandas.concat(node_deletion_results)
    node_deleteion_db.to_pickle("results/node_deletion.p")


def node_deletion_sim(year, aggregate, imports, reporters, commodities):
    results = []
    for i, comm_matrix in enumerate(imports):
        print("Running sim for comm:", commodities[i])
        # Pre-shock statistics for comm network
        curr_stats = matrix_calcs(comm_matrix)
        for j, reporter in enumerate(reporters):
            print("Running for reporter:", reporter)
            # First post-shock statistics for comm network given 'reporter' stops trading
            shocked_agg_stats, shocked_matrix_stats = node_deletion(aggregate, curr_stats, j)
            initial_loss = curr_stats['e'] - shocked_matrix_stats['e']

            # Iteration for comm network
            prev_M = copy.deepcopy(shocked_matrix_stats['matrix'])
            prev_E = e_matrix_calc(shocked_matrix_stats['alpha'], shocked_matrix_stats['beta'], prev_M)
            curr_M = iterate_matrix(prev_E, shocked_matrix_stats['m'])
            curr_E = e_matrix_calc(shocked_matrix_stats['alpha'], shocked_matrix_stats['beta'], curr_M)
            iterations = 1
            comm_rms = calc_rms(prev_E, curr_E)
            while comm_rms > 0.01 and iterations < 101:
                # print('Iteration:', iterations, "rms:", comm_rms)
                iterations += 1
                prev_E = curr_E
                curr_M = iterate_matrix(curr_E, shocked_matrix_stats['m'])
                curr_E = e_matrix_calc(shocked_matrix_stats['alpha'], shocked_matrix_stats['beta'], curr_M)
                comm_rms = calc_rms(prev_E, curr_E)

            print("Final it", iterations, "final rms;", comm_rms)
            exp_diff = curr_stats['e'] - curr_E
            amplification = sum(exp_diff) / sum(initial_loss)

            # Iteration for agg network
            agg_prev_e = shocked_agg_stats['e']
            agg_curr_M = iterate_matrix(agg_prev_e, shocked_agg_stats['m'])
            agg_curr_e = e_matrix_calc(shocked_agg_stats['alpha'], shocked_agg_stats['beta'], agg_curr_M)
            agg_iterations = 1
            agg_rms = calc_rms(agg_prev_e, agg_curr_e)
            while agg_rms > 0.01 and agg_iterations < 1001:
                # print('Agg Iteration:', agg_iterations, "rms", agg_rms)
                agg_iterations += 1
                agg_prev_e = agg_curr_e
                agg_curr_M = iterate_matrix(agg_curr_e, shocked_agg_stats['m'])
                agg_curr_e = e_matrix_calc(shocked_agg_stats['alpha'], shocked_agg_stats['beta'], agg_curr_M)
                agg_rms = calc_rms(agg_prev_e, agg_curr_e)

            print("Final agg it:", agg_iterations, "final rms", agg_rms)
            agg_exp_diff = aggregate['e'] - agg_curr_e
            agg_amplification = sum(agg_exp_diff) - sum(initial_loss)

            entry = pandas.DataFrame(
                {'year': year, 'country': reporter, 'commodity': commodities[i], 'initial_loss': sum(initial_loss),
                 'comm_iterations': iterations, 'final_comm_loss': sum(exp_diff), 'comm_amplication': amplification,
                 'agg_iterations': agg_iterations, 'final_agg_loss': agg_exp_diff,
                 'agg_amplification': agg_amplification})

            results.append(entry)

    return results


def iterate_matrix(old_exp, m):
    return numpy.matmul(numpy.diag(old_exp), m)


def node_deletion(agg_stats, comm_import_stats, reporter):
    shocked_matrix_stats = copy.deepcopy(comm_import_stats)
    shocked_agg_stats = copy.deepcopy(agg_stats)

    # Apply shock to each trade reporter 'i' has for specific commodity
    for i in range(len(shocked_matrix_stats['matrix'])):
        shocked_agg_stats['matrix'][reporter, i] -= shocked_matrix_stats['matrix'][reporter, i]
        shocked_agg_stats['matrix'][i, reporter] -= shocked_matrix_stats['matrix'][i, reporter]
        shocked_agg_stats['m'][reporter, i] -= shocked_matrix_stats['m'][reporter, i]
        shocked_agg_stats['m'][i, reporter] -= shocked_matrix_stats['m'][i, reporter]

        shocked_matrix_stats['matrix'][reporter, i] = 0
        shocked_matrix_stats['matrix'][i, reporter] = 0
        shocked_matrix_stats['m'][reporter, i] = 0
        shocked_matrix_stats['m'][i, reporter] = 0

    shocked_agg_stats['alpha'][reporter] -= shocked_matrix_stats['alpha'][reporter]
    shocked_matrix_stats['alpha'][reporter] = 0

    shocked_agg_stats['e'] = e_matrix_calc(shocked_agg_stats['alpha'], shocked_agg_stats['beta'],
                                           shocked_agg_stats['matrix'])
    shocked_matrix_stats['e'] = e_matrix_calc(shocked_matrix_stats['alpha'], shocked_matrix_stats['beta'],
                                              shocked_matrix_stats['matrix'])

    return shocked_agg_stats, shocked_matrix_stats


def calc_rms(prev_E, curr_E):
    return numpy.sqrt(((curr_E - prev_E) ** 2).mean())


def matrix_calcs(comm_matrix):
    Im = numpy.sum(comm_matrix, axis=1)
    Om = numpy.sum(comm_matrix, axis=0)
    # print(Im)
    # print(Om)
    alpha = [j / i if i and i > j else 1 for i, j in zip(Im, Om)]
    beta = [j - i if j > i else 0 for i, j in zip(Im, Om)]
    m = numpy.zeros((len(comm_matrix), len(comm_matrix)))
    for i, x in enumerate(comm_matrix):
        for j, y in enumerate(x):
            m[i, j] = y / Om[i] if Om[i] > 0 else 0

    e = e_matrix_calc(alpha, beta, m)

    return {"alpha": alpha, "beta": beta, "m": m, "e": e, 'matrix': comm_matrix}


def e_matrix_calc(alpha, beta, matrix):
    return numpy.add(
        numpy.matmul(numpy.matmul(numpy.diag(alpha), numpy.transpose(matrix)), numpy.ones(len(matrix))), beta)


def makematrices(db, partners, year):
    reporters = list(set().union(partners, pandas.unique(db.ReporterCode.values.ravel())))
    # print("Reporters:", reporters)
    reporters.sort()
    unique_entries = len(reporters)

    # print(len(list(map(lambda x: x['id'], traders))))
    commodities = pandas.unique(db.CommodityCode.values.ravel())
    commodities.sort()
    imports = [numpy.zeros((unique_entries, unique_entries)) for _ in commodities]
    aggregate = numpy.zeros((unique_entries, unique_entries))
    # exports = [numpy.zeros((unique_entries, unique_entries)) for i in commodities]

    reporter_to_i = {key: value for (value, key) in enumerate(reporters)}
    commodity_to_i = {key: value for (value, key) in enumerate(commodities)}
    # print(reporter_to_i)

    for i, reporter in enumerate(reporters):
        # i is index position in matrix
        results = db[(db['ReporterCode'] == reporter) & (db['Year'] == year)]
        if results.shape[0] == 0:
            # print("No results for code:", reporter)
            continue

        for row in results.itertuples():
            partner_code = getattr(row, 'PartnerCode')
            if partner_code == 0:
                # print("Ignoring world for reporter:", reporter)
                continue
            # print(row)
            flow = getattr(row, 'TradeFlowCode')
            flow_amount = getattr(row, 'TradeValue')
            comm_code = getattr(row, 'CommodityCode')

            partner_index = reporter_to_i[partner_code]
            comm_index = commodity_to_i[comm_code]
            matrix = imports[comm_index]
            if flow in [IMPORT, REIMPORT]:
                # print("Adding", flow_amount, "to", i, partner_index, "on", comm_index, "in imports")
                matrix[i, partner_index] = flow_amount
                aggregate[i, partner_index] = flow_amount
            else:
                # print("Adding", flow_amount, "to", i, partner_index, "on", comm_index, "in exports")
                matrix[partner_index, i] = flow_amount
                aggregate[partner_index, i] = flow_amount

    return imports, aggregate, reporters, commodities


def setupdb():
    concat = loaddb()
    if concat is not None:
        concat.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
        concat.rename(columns={'TradeValue(US$)': 'TradeValue'}, inplace=True)
        # print(concat.columns.values)
        return concat

    files = glob.glob(os.path.join(data_dir, "*.csv"))
    fields = ['Year', 'Trade Flow Code', 'Reporter Code', 'Partner Code', 'Commodity Code', 'Trade Value (US$)']
    dtypes = {'Year': int, 'Trade Flow Code': int, 'Reporter Code': int, 'Partner Code': int, 'Commodity Code': int,
              'Trade Value (US$)': int}

    dbs = []
    for i, file in enumerate(files):
        print("Trying to make", file, "no:", i)
        try:
            db = pandas.read_csv(file, usecols=fields, dtype=dtypes, low_memory=False)
            print("Made db for", file)
            dbs.append(db)
        except EmptyDataError:
            print("Unable to read", file)

    concat = pandas.concat(dbs, ignore_index=True)
    # print(concat)
    # print(concat.columns)
    # print(concat.index)
    concat.to_pickle("db.p")
    concat.rename(columns=lambda x: x.strip(), inplace=True)

    return concat


def loaddb():
    return pandas.read_pickle("db.p")


def loadcountries():
    with open(countries_file, newline='\n') as file:
        traders = json.load(file)

        return filter(lambda x: x['id'] != 'all', traders['results'])


def load_partners():
    with open(partners_file, encoding='utf-8') as partner_data:
        ignore = ['all', '0']
        partners = json.load(partner_data)

        sorted_partners = list(
            map(lambda x: int(x), list(filter(lambda x: x not in ignore, map(lambda x: x['id'], partners['results'])))))
        sorted_partners.sort()
        return sorted_partners


run()
