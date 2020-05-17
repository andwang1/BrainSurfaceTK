
import os
import sys
import atexit
import time

import mod_wsgi.server

working_directory = '/home/server/neodeepbrain/GUI'

entry_point = 'BasicSite.wsgi'
application_type = 'module'
callable_object = 'application'
mount_point = '/'
with_newrelic_agent = False
newrelic_config_file = ''
newrelic_environment = ''
reload_on_changes = False
debug_mode = False
enable_debugger = False
debugger_startup = False
enable_coverage = False
coverage_directory = ''
enable_profiler = False
profiler_directory = ''
enable_recorder = False
recorder_directory = ''
enable_gdb = False

os.environ['MOD_WSGI_EXPRESS'] = 'true'
os.environ['MOD_WSGI_SERVER_NAME'] = 'localhost'
os.environ['MOD_WSGI_SERVER_ALIASES'] = None or ''

if reload_on_changes:
    os.environ['MOD_WSGI_RELOADER_ENABLED'] = 'true'

if debug_mode:
    os.environ['MOD_WSGI_DEBUG_MODE'] = 'true'

    # We need to fiddle sys.path as we are not using daemon mode and so
    # the working directory will not be added to sys.path by virtue of
    # 'home' option to WSGIDaemonProcess directive. We could use the
    # WSGIPythonPath directive, but that will cause .pth files to also
    # be evaluated.

    sys.path.insert(0, working_directory)

if enable_debugger:
    os.environ['MOD_WSGI_DEBUGGER_ENABLED'] = 'true'

def output_coverage_report():
    coverage_info.stop()
    coverage_info.html_report(directory=coverage_directory)

if enable_coverage:
    os.environ['MOD_WSGI_COVERAGE_ENABLED'] = 'true'

    from coverage import coverage
    coverage_info = coverage()
    coverage_info.start()
    atexit.register(output_coverage_report)

def output_profiler_data():
    profiler_info.disable()
    output_file = '%s-%d.pstats' % (int(time.time()*1000000), os.getpid())
    output_file = os.path.join(profiler_directory, output_file)
    profiler_info.dump_stats(output_file)

if enable_profiler:
    os.environ['MOD_WSGI_PROFILER_ENABLED'] = 'true'

    from cProfile import Profile
    profiler_info = Profile()
    profiler_info.enable()
    atexit.register(output_profiler_data)

if enable_recorder:
    os.environ['MOD_WSGI_RECORDER_ENABLED'] = 'true'

if enable_gdb:
    os.environ['MOD_WSGI_GDB_ENABLED'] = 'true'

if with_newrelic_agent:
    if newrelic_config_file:
        os.environ['NEW_RELIC_CONFIG_FILE'] = newrelic_config_file
    if newrelic_environment:
        os.environ['NEW_RELIC_ENVIRONMENT'] = newrelic_environment

handler = mod_wsgi.server.ApplicationHandler(entry_point,
        application_type=application_type, callable_object=callable_object,
        mount_point=mount_point, with_newrelic_agent=with_newrelic_agent,
        debug_mode=debug_mode, enable_debugger=enable_debugger,
        debugger_startup=debugger_startup, enable_recorder=enable_recorder,
        recorder_directory=recorder_directory)

reload_required = handler.reload_required
handle_request = handler.handle_request

if reload_on_changes and not debug_mode:
    mod_wsgi.server.start_reloader()

