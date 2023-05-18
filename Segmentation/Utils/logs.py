import javabridge as jb


def stop_logging():
    logger_name = jb.get_static_field("org/slf4j/Logger",
                                      "ROOT_LOGGER_NAME",
                                      "Ljava/lang/String;")

    root_logger = jb.static_call("org/slf4j/LoggerFactory",
                                 "getLogger",
                                 "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                 logger_name)

    log_level = jb.get_static_field("ch/qos/logback/classic/Level",
                                    "WARN",
                                    "Lch/qos/logback/classic/Level;")

    jb.call(root_logger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            log_level)
