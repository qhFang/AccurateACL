from rlpyt.utils.logging.tabulate import tabulate
from rlpyt.utils.logging.console import mkdir_p, colorize
import numpy as np
import os
import os.path as osp
import sys
import datetime
import csv

_prefix_str = ''
_tabular_prefix_str = ''
_tabular = []
_text_fds = {}
_tabular_fds = {}  # key: file_name, value: open file
_tabular_header_written = set()
_log_tabular_only = False
_disable_prefix = False
_tf_summary_dir = None
_tf_summary_writer = None
_disabled = False
_tabular_disabled = False
_iteration = 0


def log(s, with_prefix=True, with_timestamp=True, color=None):
    if not _disabled:
        out = s
        if with_prefix and not _disable_prefix:
            out = _prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now()  # dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            # out = "%s | %s" % (timestamp, out)
        if color is not None:
            out = colorize(out, color)
        if not _log_tabular_only:
            # Also log to stdout
            print(out)
            for fd in list(_text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            sys.stdout.flush()


def record_tabular(key, val, *args, **kwargs):
    # if not _disabled and not _tabular_disabled:
    key = _tabular_prefix_str + str(key)
    _tabular.append((key, str(val)))
    if _tf_summary_writer is not None:
        _tf_summary_writer.add_scalar(key, val, _iteration)


class TerminalTablePrinter:
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()

_tabular_headers = dict()  # keys are file_names and values are the keys of the header of that tabular file


def dump_tabular(*args, **kwargs):
    if not _disabled:  # and not _tabular_disabled:
        wh = kwargs.pop("write_header", None)
        if len(_tabular) > 0:
            if _log_tabular_only:
                table_printer.print_tabular(_tabular)
            else:
                for line in tabulate(_tabular).split('\n'):
                    log(line, *args, **kwargs)
            if not _tabular_disabled:
                tabular_dict = dict(_tabular)
                # Also write to the csv files
                # This assumes that the keys in each iteration won't change!
                for tabular_file_name, tabular_fd in list(_tabular_fds.items()):
                    keys = tabular_dict.keys()
                    if tabular_file_name in _tabular_headers:
                        # check against existing keys: if new keys re-write Header and pad with NaNs
                        existing_keys = _tabular_headers[tabular_file_name]
                        if not set(existing_keys).issuperset(set(keys)):
                            joint_keys = set(keys).union(set(existing_keys))
                            tabular_fd.flush()
                            read_fd = open(tabular_file_name, 'r')
                            reader = csv.DictReader(read_fd)
                            rows = list(reader)
                            read_fd.close()
                            tabular_fd.close()
                            tabular_fd = _tabular_fds[tabular_file_name] = open(tabular_file_name, 'w')
                            new_writer = csv.DictWriter(tabular_fd, fieldnames=list(joint_keys))
                            new_writer.writeheader()
                            for row in rows:
                                for key in joint_keys:
                                    if key not in row:
                                        row[key] = np.nan
                            new_writer.writerows(rows)
                            _tabular_headers[tabular_file_name] = list(joint_keys)
                    else:
                        _tabular_headers[tabular_file_name] = keys

                    writer = csv.DictWriter(tabular_fd, fieldnames=_tabular_headers[tabular_file_name])  # list(
                    if wh or (wh is None and tabular_file_name not in _tabular_header_written):
                        writer.writeheader()
                        _tabular_header_written.add(tabular_file_name)
                        _tabular_headers[tabular_file_name] = keys
                    # add NaNs in all empty fields from the header
                    for key in _tabular_headers[tabular_file_name]:
                        if key not in tabular_dict:
                            tabular_dict[key] = np.nan
                    writer.writerow(tabular_dict)
                    tabular_fd.flush()
            del _tabular[:]


def record_tabular_misc_stat(key, values, metric=('avg', )):
    assert len(values) > 0, "Logger Error: Noting to be logged"
    if metric is None:
        assert len(values) == 1, "Logger Error: no metric specified"
        record_tabular(key, values[0])
    else:
        if 'avg' in metric:
            record_tabular(key + "Average", np.average(values))
        if 'std' in metric:
            record_tabular(key + "Std", np.std(values))
        if 'median' in metric:
            record_tabular(key + "Median", np.median(values))
        if 'min' in metric:
            record_tabular(key + "Min", np.min(values))
        if 'max' in metric:
            record_tabular(key + "Max", np.max(values))


class Logger:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()
        self._tabular_prefix_str = ''
        self._tabular = []

    def store(self, **kwargs):
        """
        Save something into the logger's current state.

        Provide an arbitrary number of keyword arguments with numerical values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
