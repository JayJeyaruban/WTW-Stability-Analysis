import requests
import json
import sys
import time

# Url for get API
api_url = "http://comtrade.un.org/api/get"

# Files
id_file = 'resources/traders.json'  # File with trader information
class_file = 'resources/classification.json'  # File for item classifications. Possibly redundant
partners_file = 'resources/partners.json'  # File with trading partner information

WAIT_TIME = 1000 * 60 * 0.6
PARTNER_QUERY_BLOCK = 5  # Max number of partners to query with. No limit indicator in API doc however web UI allows 5
QUERY_BLOCK_REDUCE = 3

# Query params
MAX_RETURN = '50000'
TYPE = 'C'  # C for commodities, S for services
CC = 'AG2'  # See API docs (https://comtrade.un.org/data/doc/api/#DataRequests) for full info
FREQ = 'Y'  # M for monthly, Y for yearly
PX = 'HS'   # See API docs (https://comtrade.un.org/data/doc/api/#DataRequests) for full info
FMT = 'csv'  # csv or json
YEARS = range(2000, 2017)
SKIP_ID = []


class ReqMaker:
    def __init__(self):
        self.last_call = self.__gettime()
        print 'Reqmaker initialized'

    @staticmethod
    def __gettime():
        return int(round(time.time() * 1000))

    # Performs http request
    def makereq(self, apiurl, parameters):
        current_time = self.__gettime()
        towait = WAIT_TIME - (current_time - self.last_call)
        if towait > 0:
            print 'Waiting ', towait / 1000, ' seconds'
            time.sleep(towait / 1000)

        print "Making request"
        r = requests.get(apiurl, params=parameters)
        print 'Url: ', r.url

        if r.ok:
            print "Response OK"
            print r.status_code
            self.last_call = self.__gettime()

        elif r.status_code == 409:
            print 'Calls too frequent, waiting...'
            time.sleep(5)

        else:
            print "Response not OK ", r.status_code
            print "Failed call: ", r.url
            print 'Dump: ', r.content

        return r


def run():
    traders = load_traders()
    # print ids

    partners = load_partners()
    # print partners

    reqmaker = ReqMaker()
    for trader in traders:
        if int(trader['id']) in SKIP_ID:
            continue
        # if trader['id'] != '826':
        #     continue

        savedata(makefilenamefromtrader(trader), getdata(reqmaker, trader, partners))


def makefilenamefromtrader(trader):
    return trader['id'] + '-' + trader['text'] + '.csv'


def getdata(reqmaker, trader, partners):
    first = True
    for year in YEARS:
        # og_partners = partners.remove(int(trader['id']))
        og_partners = partners.remove(int(trader['id']))
        query_partners = og_partners[:PARTNER_QUERY_BLOCK]
        og_partners = og_partners[PARTNER_QUERY_BLOCK:]

        results = ''
        while len(og_partners) > 0 | len(query_partners) > 0:
            fakepartners = [4, 8]
            # print og_partners
            payload = makepayload(trader, query_partners, year)
            # print payload

            r = reqmaker.makereq(api_url, payload)
            if r.ok:
                print 'Successful query; code: ', r.status_code
                query_partners = og_partners[:PARTNER_QUERY_BLOCK]
                og_partners = og_partners[PARTNER_QUERY_BLOCK:]
                # print 'Next og: ', og_partners
                # print 'Next query partners: ', query_partners

                if first:
                    results = results + r.content
                    first = False
                else:
                    results = results + removecsvtitleline(r.content)
            elif r.status_code == 5004:
                print 'Too many entries, reducing'
                og_partners = query_partners[len(query_partners) - QUERY_BLOCK_REDUCE:] + og_partners
                query_partners = query_partners[:QUERY_BLOCK_REDUCE]
            else:
                sys.exit(1)

    return results


def removecsvtitleline(csv_string):
    nli = csv_string.index('\n')
    return csv_string[(nli + 1):]


def makepayload(trader, partners, year):
    print 'Making payload for ', trader, 'with year ', year, 'and partners ', partners
    formatted_partners = ''
    for partner in partners:
        if formatted_partners == '':
            formatted_partners = str(partner)
        else:
            formatted_partners = formatted_partners + ',' + str(partner)

    return {'r': trader['id'], 'freq': FREQ, 'ps': year, 'px': PX, 'p': formatted_partners, 'cc': CC,
            'fmt': FMT, 'max': MAX_RETURN, 'type': TYPE}


def savedata(filename, data):
    # print data
    print 'Saving to file: ', filename
    with open('output/' + filename, 'w') as f:
        print >> f, data


def load_traders():
    with open(id_file) as trader_data:
        traders = json.load(trader_data)

        return filter(lambda x: x['id'] != 'all', traders['results'])


def load_partners():
    with open(partners_file) as partner_data:
        ignore = ['all', '0']
        partners = json.load(partner_data)

        sorted_partners = filter(lambda x: x not in ignore, map(lambda x: x['id'], partners['results']))
        sorted_partners.sort(key=float)
        return sorted_partners


run()
