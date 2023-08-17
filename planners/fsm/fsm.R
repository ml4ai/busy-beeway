add_state.FSM <- function(fsm,name,handler) {
    name <- toupper(name)
    fsm$handlers[[name]] <- handler
    fsm
}

add_state <- function(fsm,name,handler) {
    UseMethod("add_state")
}

set_start.FSM <- function(fsm,name) {
    fsm$currentState <- toupper(name)
    fsm
}

set_start <- function(fsm,name) {
    UseMethod("set_start")
}

transition.FSM <- function(fsm,input) {
    if (is.null(fsm$currentState)) {
        stop("try calling set_start(FSM,...) first!")
    }
    if (is.null(fsm$handlers[[toupper(fsm$currentState)]])) {
        cat("Terminal State has been reached, cannot transition!\n")
    }
    else {
        fsm$currentState <- toupper(fsm$handlers[[toupper(fsm$currentState)]](input))
        #cat("reached",fsm$currentState,"\n")
    }
    fsm
}

transition <- function(fsm,input) {
    UseMethod("transition")
}