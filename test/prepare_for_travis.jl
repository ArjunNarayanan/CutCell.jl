function comment_using_revise(filename)
    lines = readlines(filename)
    idx = findfirst(lines .== "using Revise")
    if isnothing(idx)
        @warn "Could not find \"using Revise\" in file $filename, doing nothing."
    else
        lines[idx] = "# using Revise"
        open(filename, "w") do f
            for l in lines
                println(f,l)
            end
        end
    end
end

testfiles = filter(x->startswith(x,"test_"),readdir("test"))
testfiles = ["test/"*s for s in testfiles]
for f in testfiles
    comment_using_revise(f)
end
