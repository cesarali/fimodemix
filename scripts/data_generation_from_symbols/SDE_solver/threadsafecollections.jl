module ThreadSafeCollections

using Base.Threads: ReentrantLock

export ThreadSafeDict, ThreadSafeSet, ThreadSafeList, add!, get, remove!, length


struct ThreadSafeDict{K,V}
    dict::Dict{K,V}
    lock::ReentrantLock
end

function ThreadSafeDict{K,V}() where {K,V}
    ThreadSafeDict{K,V}(Dict{K,V}(), ReentrantLock())
end

function add!(tsd::ThreadSafeDict, key, value)
    lock(tsd.lock) do
        tsd.dict[key] = value
    end
end

function get(tsd::ThreadSafeDict, key)
    lock(tsd.lock) do
        return tsd.dict[key]
    end
end

function remove!(tsd::ThreadSafeDict, key)
    lock(tsd.lock) do
        delete!(tsd.dict, key)
    end
end

function length(tsd::ThreadSafeDict)
    lock(tsd.lock) do
        return Base.length(tsd.dict)
    end
end


struct ThreadSafeSet{T}
    set::Set{T}
    lock::ReentrantLock
end

function ThreadSafeSet{T}() where {T}
    ThreadSafeSet{T}(Set{T}(), ReentrantLock())
end

function add!(tss::ThreadSafeSet, item)
    lock(tss.lock) do
        push!(tss.set, item)
    end
end

function contains(tss::ThreadSafeSet, item)
    lock(tss.lock) do
        return item in tss.set
    end
end

function remove!(tss::ThreadSafeSet, item)
    lock(tss.lock) do
        delete!(tss.set, item)
    end
end

function length(tss::ThreadSafeSet)
    lock(tss.lock) do
        return Base.length(tss.set)
    end
end

struct ThreadSafeList{T}
    list::Vector{T}
    lock::ReentrantLock
end

function ThreadSafeList{T}() where {T}
    ThreadSafeList{T}(Vector{T}(), ReentrantLock())
end

function add!(tsl::ThreadSafeList, item)
    lock(tsl.lock) do
        push!(tsl.list, item)
    end
end

function get(tsl::ThreadSafeList, index::Int)
    lock(tsl.lock) do
        return tsl.list[index]
    end
end

function remove!(tsl::ThreadSafeList, index::Int)
    lock(tsl.lock) do
        deleteat!(tsl.list, index)
    end
end

function length(tsl::ThreadSafeList)
    lock(tsl.lock) do
        return length(tsl.list)
    end
end

end