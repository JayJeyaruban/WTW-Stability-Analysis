import networkx
import json
import pandas
from pandas.errors import EmptyDataError
import glob
import os
import matplotlib.pyplot as pyplot
import numpy
from numpy.linalg import inv
import copy

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

    for year in YEARS:
        imports, aggregate, reporters, comm_map = makematrices(db, partners, year)
        # print(imports)
        # print(exports)
        agg = matrix_calcs(aggregate)

        node_deletion_sim(agg, imports, reporters)


def node_deletion_sim(aggregate, imports, reporters):
    for i, comm_matrix in enumerate(imports):
        # Pre-shock statistics for comm network
        curr_stats = matrix_calcs(comm_matrix)
        for reporter in list(range(0, len(reporters))):
            # First post-shock statistics for comm network given 'reporter' stops trading
            shocked_matrix_stats = node_deletion(curr_stats, reporter)

            # Next iteration
            prev_M = copy.deepcopy(shocked_matrix_stats['matrix'])
            prev_E = e_matrix_calc(shocked_matrix_stats['alpha'], shocked_matrix_stats['beta'], prev_M)
            curr_M = iterate_matrix(prev_E, shocked_matrix_stats['m'])
            curr_E = e_matrix_calc(shocked_matrix_stats['alpha'], shocked_matrix_stats['beta'], curr_M)
            iterations = 1
            while calc_rms(prev_E, curr_E) > 0.01 and iterations < 101:
                print('Iteration:', iterations, "score:", calc_rms(prev_E, curr_E))
                iterations += 1
                prev_E = curr_E
                curr_M = iterate_matrix(curr_E, shocked_matrix_stats['m'])
                curr_E = e_matrix_calc(shocked_matrix_stats['alpha'], shocked_matrix_stats['beta'], curr_M)


def iterate_matrix(old_exp, m):
    return numpy.matmul(numpy.diag(old_exp), m)


def node_deletion(comm_import_stats, reporter):
    shocked_matrix_stats = copy.deepcopy(comm_import_stats)

    # Apply shock to each trade reporter has for specific commodity
    for i in range(len(shocked_matrix_stats['matrix'])):
        shocked_matrix_stats['matrix'][reporter, i] = 0
        shocked_matrix_stats['matrix'][i, reporter] = 0
        shocked_matrix_stats['m'][reporter, i] = 0
        shocked_matrix_stats['m'][i, reporter] = 0

    shocked_matrix_stats['alpha'][reporter] = 0

    shocked_matrix_stats['e'] = e_matrix_calc(shocked_matrix_stats['alpha'], shocked_matrix_stats['beta'],
                                              shocked_matrix_stats['matrix'])

    return shocked_matrix_stats


def calc_rms(prev_E, curr_E):
    return numpy.sqrt(((curr_E - prev_E) ** 2).mean())


def matrix_calcs(comm_matrix):
    Im = numpy.sum(comm_matrix, axis=1)
    Om = numpy.sum(comm_matrix, axis=0)
    # print(Im)
    # print(Om)
    alpha = [i / j if j and i >= j else 1 for i, j in zip(Im, Om)]
    beta = [j - i if j > i else 0 for i, j in zip(Im, Om)]
    m = numpy.zeros((len(comm_matrix), len(comm_matrix)))
    for i, x in enumerate(comm_matrix):
        for j, y in enumerate(x):
            m[i, j] = y / Om[i] if Om[i] > 0 else 0

    # TODO Check this formula
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
    imports = [numpy.zeros((unique_entries, unique_entries)) for i in commodities]
    aggregate = numpy.zeros((unique_entries, unique_entries))
    # exports = [numpy.zeros((unique_entries, unique_entries)) for i in commodities]

    reporter_to_i = {key: value for (value, key) in enumerate(reporters)}
    commodity_to_i = {key: value for (value, key) in enumerate(commodities)}
    # print(reporter_to_i)

    for i, reporter in enumerate(reporters):
        # i is index position in matrix
        results = db[(db['ReporterCode'] == reporter) & (db['Year'] == year)]
        if results.shape[0] == 0:
            print("No results for code:", reporter)
            continue

        for row in results.itertuples():
            partner_code = getattr(row, 'PartnerCode')
            if partner_code == 0:
                print("Ignoring world for reporter:", reporter)
                continue
            # print(row)
            flow = getattr(row, 'TradeFlowCode')
            flow_amount = getattr(row, 'TradeValue')
            comm_code = getattr(row, 'CommodityCode')

            partner_index = reporter_to_i[partner_code]
            comm_index = commodity_to_i[comm_code]
            matrix = imports[comm_index]
            if flow in [IMPORT, REIMPORT]:
                print("Adding", flow_amount, "to", i, partner_index, "on", comm_index, "in imports")
                matrix[i, partner_index] = flow_amount
                aggregate[i, partner_index] = flow_amount
            else:
                print("Adding", flow_amount, "to", i, partner_index, "on", comm_index, "in exports")
                matrix[partner_index, i] = flow_amount
                aggregate[partner_index, i] = flow_amount

    return imports, aggregate, reporters, commodity_to_i


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


def setupgraphs(db, countries):
    # print(db)
    graphs = []
    for year in range(2000, 2017):
        graph = networkx.DiGraph()
        for country in countries:
            id = int(country['id'])
            result = db[(db['ReporterCode'] == id) & (db['Year'] == year)]
            if result.shape[0] == 0:
                print("No results")
                continue

            print(result)

            addtograph(graph, id, result.itertuples())

        graphs.append((year, graph))

    return graphs


def addtograph(graph, country_code, trade_data):
    if not graph.has_node(country_code):
        graph.add_node(country_code)

    for row in trade_data:
        # print(row)
        partner = getattr(row, "PartnerCode")
        if partner == 0:
            continue

        if not graph.has_node(partner):
            graph.add_node(partner)

        flow = getattr(row, 'TradeFlowCode')
        flow_amount = getattr(row, 'TradeValue')
        comm_code = getattr(row, 'CommodityCode')

        edge = tradeflowswitch(country_code, partner, flow)
        if edge == -1:
            print("Handle invalid flow", flow)

        edge
        print(edge)
        if graph.has_edge(*edge):
            print("To handle")

        graph.add_weighted_edges_from(edge + ({'weight': flow_amount, 'comm': comm_code},))


def drawgraphs(graphs):
    for i, graph in enumerate(graphs):
        pos = networkx.layout.spring_layout(graph)

        nodesize = [3 + 10 * i for i in range(len(graph))]
        M = graph.number_of_edges()
        edge_colors = range(2, M + 2)
        edge_alphas = [(5 + 1 / M + 4) for i in range(M)]

        nodes = networkx.draw_networkx_nodes(graph, pos, node_size=nodesize, node_color='blue')
        edges = networkx.draw_networkx_edges(graph, pos, node_size=nodesize, arrowstyle='->', arrowsize=10,
                                             edge_colors=edge_colors, edge_cmap=pyplot.cm.Blues, width=2)

        for i in range(M):
            edges[i].set_alpha(edge_alphas[i])

        ax = pyplot.gca()
        ax.set_axis_off()
        pyplot.savefig(i + '.png')


def tradeflowswitch(country1, country2, flow):
    return {
        IMPORT: (country2, country1),
        EXPORT: (country1, country2),
        REIMPORT: (country2, country1),
        REEXPORT: (country1, country2)
    }.get(flow, -1)


def loadcountries():
    with open(countries_file, newline='\n') as file:
        traders = json.load(file)

        return filter(lambda x: x['id'] != 'all', traders['results'])


def load_partners():
    with open(partners_file) as partner_data:
        ignore = ['all', '0']
        partners = json.load(partner_data)

        sorted_partners = list(
            map(lambda x: int(x), list(filter(lambda x: x not in ignore, map(lambda x: x['id'], partners['results'])))))
        sorted_partners.sort()
        return sorted_partners


run()
