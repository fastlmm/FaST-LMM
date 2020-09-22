from __future__ import absolute_import
import pandas as pd
import logging
import six

class NearBronze:

    @staticmethod
    def Parse():
        import argparse
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        # required arguments
        parser.add_argument('bim_filename', help='Input Bim file', type=str)
        parser.add_argument('bronzelist_filename', help='List of bronze snps', type=str)
        parser.add_argument('output_filename', help='Output Bim', type=str)
        
        # required optional arguments
        parser.add_argument('-distance', help='distance', type=float, default=None)
        parser.add_argument('-position', help='position', type=int, default=None)
        parser.add_argument('-log', help='flag to control verbosity of logging', type=str, default="INFO")
        
        # parse arguments
        args = parser.parse_args()

        if bool(args.distance == None) == bool(args.position == None) : raise Exception("Should give 'distance' or 'position', but not both")

        return args

    @staticmethod
    def Set_Up_Logging(args):
        numeric_level = getattr(logging, args.log.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % args.log)
        logging.basicConfig(level=numeric_level)

    #@staticmethod
    #def Bronze_Rows(bronzefields, snp, bim_chrom, bim_value):
    #    try:
    #        item_or_list = bronzefields.loc[snp] #return either the row or list of rows with this snp
    #        return item_or_list, "1"
    #    except KeyError:
    #        item_or_list = None
        
    #    #Find the closest snp
    #    bestsnp=None
    #    bestdiff = float("inf")
    #    bestitem_or_list = None
    #    for bronze_snp in bronzefields.index:
    #        item_or_list = bronzefields.loc[bronze_snp]
    #        bronze_chrom, bronze_value_pair = NearBronze.ExtraFromBronzeRow(item_or_list, bronze_snp)
    #        if bronze_chrom == bim_chrom:
    #            for bronze_value in bronze_value_pair:
    #                diff = abs(bronze_value-bim_value)
    #                if diff < bestdiff:
    #                    bestdiff = diff
    #                    bestsnp = bronze_snp
    #                    bestitem_or_list = item_or_list


    #    return bestitem_or_list, "0"

    #@staticmethod
    #def IsSingleton(bronze_chrom):
    #    try:
    #        length = len(bronze_chrom)
    #        return True
    #    except:
    #        return False

    #@staticmethod
    #def ExtraFromBronzeRow(bronze_rows, snp):
    #    bronze_chrom = bronze_rows['chr']
    #    bronze_value_pair = bronze_rows.values[1]
        
    #    if NearBronze.IsSingleton(bronze_chrom):
    #        if len(bronze_chrom) != 2 : raise Exception("Expect at most two end points (snp={0})".format(snp))
    #        if bronze_chrom[0] != bronze_chrom[1] : raise Exception("Expect both end points to be on same chrom (snp={0})".format(snp))
    #        if bronze_value_pair[0] > bronze_value_pair[1] : raise Exception("Expect first end point to be no more than second end point  (snp={0})".format(snp))
    #        bronze_chrom = bronze_chrom[0]
    #    else:
    #        bronze_value_pair = [bronze_value_pair, bronze_value_pair]
        
        
    #    return bronze_chrom, bronze_value_pair

    #    return allowed_diff, bim_value


    @staticmethod
    def Extract_Distance(args):
        if args.distance:
            distance_index = 2 #genetic_distance
            allowed_diff  = args.distance
        else:
            distance_index = 3
            allowed_diff  = args.position
        return allowed_diff, distance_index

    @staticmethod #same as in fastlmm utils
    def create_directory_if_necessary(name, isfile=True):    
        import os
        if isfile:
            directory_name = os.path.dirname(name)
        else:
            directory_name = name

        if directory_name != "":
            try:
                os.makedirs(directory_name)
            except OSError as e:
                if not os.path.isdir(directory_name):
                    raise Exception("not valid path: " + directory_name)


    @staticmethod
    def CreateBronzeChromToSortedValueAndSnpList(args, chromToSnpToValue):
        bronzeChromToSortedValueAndSnpList = {}
        with open(args.bronzelist_filename) as file:
            header = next(file,None)
            if header is None or len(header.split("\t")) != 3 : raise Exception("Expect bronze file to have three-column header line")
            for line in file:
                row = line.strip().split("\t")  # e.g. ['rs6684865', '1', '2553624']
                snp = row[0]
                chrom = row[1]
                value = chromToSnpToValue.get(chrom,{}).get(snp,None)
                if value is None:
                    logging.warning("snp {0} appears in the bronze list, but not the bim file".format(snp))
                else:
                    valueAndSnpList = bronzeChromToSortedValueAndSnpList.setdefault(chrom,[])
                    valueAndSnpList.append((value, snp))
        chromToSortedValuesAndSnps = {}
        for chrom, valueAndSnpList in six.iteritems(bronzeChromToSortedValueAndSnpList):
            sortedList = sorted(valueAndSnpList,key=lambda valueAndSnp: valueAndSnp[0])
            sortedValues = [valueAndSnp[0] for valueAndSnp in sortedList]
            sortedSnps = [valueAndSnp[1] for valueAndSnp in sortedList]
            chromToSortedValuesAndSnps[chrom] = (sortedValues, sortedSnps)
        
        return chromToSortedValuesAndSnps

    @staticmethod
    def CreateChromToSnpToValue(args, distance_index):
        chromToSnpToValue = {}
        with open(args.bim_filename,) as bim_file:
            for bimindex, line in enumerate(bim_file):
                if bimindex % 10000 == 0 :
                    logging.info("First time reading bim line {0}".format(bimindex))
                linestriped = line.strip()
                row = linestriped.split()  # e.g. ['1', 'rs3094315', '0.830874', '752566', '2', '1']
                chrom = row[0]
                snp = row[1]
                value = float(row[distance_index])
                snpToValue = chromToSnpToValue.setdefault(chrom,{})
                snpToValue[snp] = value
        
        
        
        
        return chromToSnpToValue

    @staticmethod
    def main():
        args = NearBronze.Parse()

        NearBronze.Set_Up_Logging(args)

        allowed_diff, distance_index = NearBronze.Extract_Distance(args)

        chromToSnpToValue = NearBronze.CreateChromToSnpToValue(args, distance_index)



        bronzeChromToSortedValueAndSnpLists = NearBronze.CreateBronzeChromToSortedValueAndSnpList(args, chromToSnpToValue)


        NearBronze.create_directory_if_necessary(args.output_filename)
        with open(args.bim_filename,) as bim_file:
            with open(args.output_filename, mode='w') as output_file:
                for bimindex, line in enumerate(bim_file):
                    if bimindex % 10000 == 0 :
                        logging.info("Writing bim line {0}".format(bimindex))
                    linestriped = line.strip()
                    row = linestriped.split()  # e.g. ['1', 'rs3094315', '0.830874', '752566', '2', '1']
                    chrom = row[0]
                    snp = row[1]
                    value = float(row[distance_index])

                    sortedValues, sortedSnps = bronzeChromToSortedValueAndSnpLists.get(chrom,([],[]))
                    import bisect
                    hiIndex = bisect.bisect_right(sortedValues, value)
                    loIndex = hiIndex - 1
                    if loIndex < 0 :
                        loValue = float("-inf")
                        loSnp = None
                    else:
                        if len(sortedValues) == 0 : raise Exception("assert")
                        loValue = sortedValues[loIndex]
                        loSnp = sortedSnps[loIndex]

                    if hiIndex == len(sortedValues):
                        hiValue = float("inf")
                        hiSnp = None
                    else:
                        if len(sortedValues) == 0 : raise Exception("assert")
                        hiValue = sortedValues[hiIndex]
                        hiSnp = sortedSnps[hiIndex]

                    if value < loValue or value > hiValue : raise Exception("assert")

                    if loSnp is not None and loSnp == hiSnp: #we are in a snp range
                        isClose = True
                        bronzeSnp = loSnp
                    else: # not in a range
                        if abs(value - loValue) < abs(value - hiValue):
                            isClose = abs(value - loValue) < allowed_diff
                            bronzeSnp = loSnp
                        else:
                            isClose = abs(value - hiValue) < allowed_diff
                            bronzeSnp = hiSnp

                    outline ="{0}\t{1}\t{2}\n".format(linestriped, (0,1)[bronzeSnp == snp], (0,1)[isClose])
                    output_file.write(outline)
                


    #@staticmethod
    #def main():
    #    args = NearBronze.Parse()

    #    NearBronze.Set_Up_Logging(args)

    #    allowed_diff, distance_index = NearBronze.Extract_Distance(args)

    #    bimfields = pd.read_csv(args.bim_filename,delimiter = '\t',header=None,index_col=False)
    #    bronzefields = pd.read_csv(args.bronzelist_filename,delimiter = '\t',index_col=0)
    #    bronze_snps_not_seen = set(bronzefields.index)

    #    NearBronze.create_directory_if_necessary(args.output_filename)
    #    with open(args.output_filename, mode='w') as f:

    #        for bimindex, bimrow in enumerate(bimfields.values):
    #            if bimindex % 1000 == 0 :
    #                logging.info("Writing bim line {0}".format(bimindex))
    #            f.write("\t".join([str(x) for x in bimrow]))

    #            bim_chrom = bimrow[0]
    #            snp = bimrow[1]
    #            bim_value = bimrow[distance_index]

    #            bronze_rows, match_char = NearBronze.Bronze_Rows(bronzefields, snp, bim_chrom, bim_value)

    #            if bronze_rows is None:
    #                f.write("\t0\t0\n") #nothing close found
    #            else:
    #                f.write("\t" + match_char)  #closest snp found in bronze list
    #                bronze_snps_not_seen.discard(snp)
    #                bronze_chrom, bronze_value_pair = NearBronze.ExtraFromBronzeRow(bronze_rows, snp)
    #                if bim_chrom != bronze_chrom :
    #                    f.write("\t0\n") # in range #chrom is difference, so can't match
    #                else :
    #                    if bronze_value_pair[0] - allowed_diff <= bim_value <= bronze_value_pair[1] + allowed_diff:
    #                        f.write("\t1\n") # in range
    #                    else:
    #                        f.write("\t0\n") # not in range


    #    if bronze_snps_not_seen:
    #        logging.warning("These snps appear in the bronze list but were not seen in the bim file: {0}".format(", ".join(bronze_snps_not_seen)))

if __name__ == "__main__":
    NearBronze.main()