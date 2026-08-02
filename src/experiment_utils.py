def format_value_for_filename(value):
    """
    Convert parameter values to filename-safe strings.

    Examples:
        0.5  -> 0p5
        2.0  -> 2
        -3   -> m3
    """

    if isinstance(value, float):
        if value.is_integer():
            value = int(value)

    return str(value).replace(".", "p").replace("-", "m")


def build_experiment_tag(num_servers,
                         arrival_model,
                         arrival_scale_C,
                         arrival_alpha,
                         setup_time,
                         simulation_time,
                         turn_on_mode,
                         response_method,
                         num_seeds):
    """
    Build a filename tag that describes the main experimental setting.

    The returned tag is used to avoid overwriting CSV files when running
    different parameter settings.
    """

    tag = (
        f"n{format_value_for_filename(num_servers)}"
        f"_{arrival_model}"
        f"_C{format_value_for_filename(arrival_scale_C)}"
        f"_alpha{format_value_for_filename(arrival_alpha)}"
        f"_setup{format_value_for_filename(setup_time)}"
        f"_T{format_value_for_filename(simulation_time)}"
        f"_{turn_on_mode}"
        f"_{response_method}"
        f"_seed{format_value_for_filename(num_seeds)}"
    )

    return tag