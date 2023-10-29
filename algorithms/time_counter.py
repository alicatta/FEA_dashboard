import time

def convert_timer_to_readable(secs):
    mins, s = divmod(secs, 60)
    h, mins = divmod(mins, 60)

    h_str = f'{int(h)}h ' if h > 0 else ''
    mins_str = f'{int(mins)}m ' if mins > 0 else ''
    s_str = f'{round(s, 2)}s'

    return h_str + mins_str + s_str

def get_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    return convert_timer_to_readable(elapsed_time)