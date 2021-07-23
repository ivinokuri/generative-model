#!/usr/bin/env python
import json
import inspect
import os

class ShrinkRules:
    def __init__(self, rules):

        self.result = self.shrink_file(rules)
        self.new_save_rules_learning_to_file(self.result)
        # self.cols_digits = self.get_value_by_topic(rules, "same digits number")
        # self.pos_columns = self.get_list_by_topic(rules, "positive values")
        # self.neg_columns = self.get_list_by_topic(rules, "negative values")
        # self.not_neg_columns = self.get_list_by_topic(rules, "not negative values")
        # self.one_value_columns = self.get_value_by_topic(rules, "exactly one")
        # self.cov_percent_columns = self.get_value_by_topic(rules, "coverage percentage columns")
        # self.corr_columns = self.get_corr_topics(rules, "corresponding columns")

    def shrink_file(self, rules):
        for val in rules.keys():
            rules[val] = self.check_best_result(rules[val])
        return rules

    def check_best_result(self, key):
        positive_value = False
        negative_value = False
        not_negative_value = False
        # exactly_one = False
        # exactly_one_item = None
        same_digits = False
        same_digits_item = None
        same_digits_sign = -2
        coverage_percentage_columns = False
        coverage_percentage_columns_item = None
        for key_val in key.keys():
            if key_val == 'coverage percentage columns':
                coverage_percentage_columns = True
                coverage_percentage_columns_item = key[key_val]
            if key_val == 'same digits number':
                same_digits = True
                same_digits_item = key[key_val]
                if 'positive values' in key:
                    same_digits_sign = 1
                elif 'negative values' in key:
                    same_digits_sign = -1
                elif 'not negative values' in key:
                    same_digits_sign = 0
            # if key_val == 'exactly one':
            #     exactly_one = True
            #     exactly_one_item = key[key_val]
            if key_val == 'positive values':
                positive_value = True
            if key_val == 'negative values':
                negative_value = True
            if key_val == 'not negative values':
                not_negative_value = True
        result = {}
        # if exactly_one:
        #     result['exactly one'] = exactly_one_item
        # el
        if coverage_percentage_columns:
            result['coverage percentage columns'] = coverage_percentage_columns_item
        elif same_digits:
            result['bound'] = self.new_format(same_digits_item, same_digits_sign)
        elif positive_value:
            result['positive values'] = 1
        elif negative_value:
            result['negative values'] = -1
        elif not_negative_value:
            result['not negative values'] = 0
        return result

    def new_format(self, num_of_dig, sign):
        result = {}
        low = 0
        high = 0
        abs_item = 0
        if num_of_dig == 1:
            if sign == 1:
                low = 1
                high = 10
            elif sign == -1:
                low = -10
                high = 0
            elif sign == 0:
                low = 0
                high = 10
            else:
                low = -10**num_of_dig
                high = 10**num_of_dig
        elif num_of_dig > 1:
            if sign == 1 or sign == 0:
                low = 10**num_of_dig-1
                high = 10**num_of_dig
            elif sign == -1:
                low = (-10**num_of_dig)+1
                high = (-10**num_of_dig-1)+1
            else:
                low = 10 ** num_of_dig - 1
                high = 10 ** num_of_dig
                abs_item = 1
        result['low'] = low
        result['high'] = high
        result['abs'] = abs_item
        return result

    def check_best_predicate(self, *predicate):
        print predicate

    def get_value_by_topic(self,rules, key):
        ret = {}
        for val in rules.keys():
            for key_val in rules[val].keys():
                if key_val == key:
                    result = {}
                    result = rules[val][key_val]
                    ret[val.encode("utf-8")] = result
        return ret


    def get_list_by_topic(self, rules, key):
        topic_list = []
        for val in rules.keys():
            for key_val in rules[val].keys():
                if key_val == key:
                    topic_list.append(val.encode("utf-8"))
        return topic_list


    def get_corr_topics(self, rules, key):
        ret = {}
        for val in rules.keys():
            for key_val in rules[val].keys():
                if key_val == key:
                    for key_val_inside in rules[val][key_val].keys():
                        re = {}
                        for key_val_inside_result in rules[val][key_val][key_val_inside].keys():
                            re[key_val_inside_result.encode("utf-8")] = rules[val][key_val][key_val_inside][
                                key_val_inside_result]
                        ret[(val.encode("utf-8"), key_val_inside.encode("utf-8"))] = re
        return ret

    def new_save_rules_learning_to_file(self, rules_stats):
        filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/rules_shrink.json"
        f = open(filepath, "w")
        json.dump(rules_stats, f)

if __name__ == '__main__':
    with open('rules.json', "rb") as f:
        raw_rules = f.read()
    rules = json.loads(raw_rules)
    shrink_rules = ShrinkRules(rules)
    print "finished"

    # import os
    #
    # if os.path.exists("rules_arrange.json"):
    #     os.remove("rules_arrange.json")
    # if os.path.exists("rules_shrink_arrange.json"):
    #     os.remove("rules_shrink_arrange.json")
    #
    # with open('rules.json', "rb") as f1:
    #     raw_rules = f1.read()
    # rules1 = json.loads(raw_rules)
    # for val in rules1:
    #     with open('rules_arrange.json', "a") as f2:
    #         topic = str(val.encode("utf-8")) + " " + str(rules1[val.encode("utf-8")])
    #         f2.write(topic + "\n")
    #         f2.close()
    # with open('rules_shrink.json', "rb") as f3:
    #     raw_rules = f3.read()
    # rules2 = json.loads(raw_rules)
    # for val in rules2:
    #     with open('rules_shrink_arrange.json', "a") as f4:
    #         f4.writelines((str(val) + " " + str(rules2[val])).encode("utf-8"))
    #         f4.close()

