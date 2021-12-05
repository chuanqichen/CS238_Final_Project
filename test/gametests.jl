using Test

include("../src/game.jl")

@testset "2048 symmetries testing" begin    
    @testset "rot90 actions mapping" begin
        board::Bitboard = initbboard()
        for i in 1:10
            board = move(board, Dirs(rand(0:3)))
            board = add_tile(board)
        end
        display(board)
        board = board |> rotl90
        display(board)
    end
    
end

