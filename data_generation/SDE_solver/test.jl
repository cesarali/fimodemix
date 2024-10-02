function repeat_forever(i, timer=Timer(1.0))
    while true
        isopen(timer) || return "$i: timer closed!"
        yield()

    end

    return "$timer finished!"
end

Threads.@threads for i in 1:10
    print(repeat_forever(i, Timer(i)))
end