module LibRandom

export DSFMT_state, dsfmt_get_min_array_size, dsfmt_get_idstring,
       dsfmt_init_gen_rand, dsfmt_gv_init_gen_rand, 
       dsfmt_init_by_array, dsfmt_gv_init_by_array,
       dsfmt_genrand_close1_open2, dsfmt_gv_genrand_close1_open2,
       dsfmt_genrand_close_open, dsfmt_gv_genrand_close_open, 
       dsfmt_genrand_open_close, dsfmt_gv_genrand_open_close, 
       dsfmt_genrand_open_open, dsfmt_gv_genrand_open_open, 
       dsfmt_fill_array_close1_open2!, dsfmt_gv_fill_array_close1_open2!,
       dsfmt_fill_array_close_open!, dsfmt_gv_fill_array_close_open!, 
       dsfmt_fill_array_open_close!, dsfmt_gv_fill_array_open_close!, 
       dsfmt_fill_array_open_open!, dsfmt_gv_fill_array_open_open!, 
       dsfmt_genrand_uint32, dsfmt_gv_genrand_uint32, 
       randmtzig_randn, randmtzig_fill_randn!, 
       randmtzig_gv_randn, randmtzig_gv_fill_randn!,
       randmtzig_exprnd, randmtzig_fill_exprnd!,
       win32_SystemFunction036!

## DSFMT

type DSFMT_state
    val::Vector{Int32}
    DSFMT_state() = new(Array(Int32, 770))
end

function dsfmt_get_idstring()
    idstring = ccall((:dsfmt_get_idstring,:librandom),
                     Ptr{Uint8},
                     ())
    return bytestring(idstring)
end

function dsfmt_get_min_array_size()
    min_array_size = ccall((:dsfmt_get_min_array_size,:librandom), 
                           Int32, 
                           ())
end

const dsfmt_min_array_size = dsfmt_get_min_array_size()

function dsfmt_init_gen_rand(s::DSFMT_state, seed::Uint32)
    ccall((:dsfmt_init_gen_rand,:librandom),
          Void, 
          (Ptr{Void}, Uint32,), 
          s.val, seed)
end

function dsfmt_gv_init_gen_rand(seed::Uint32)
    ccall((:dsfmt_gv_init_gen_rand,:librandom),
          Void,
          (Uint32,),
          seed)
end

function dsfmt_init_by_array(s::DSFMT_state, seed::Vector{Uint32})
    ccall((:dsfmt_init_by_array,:librandom),
          Void, 
          (Ptr{Void}, Ptr{Uint32}, Int32), 
          s.val, seed, length(seed))
end

function dsfmt_gv_init_by_array(seed::Vector{Uint32})
    ccall((:dsfmt_gv_init_by_array,:librandom),
        Void, 
        (Ptr{Uint32}, Int32), 
        seed, length(seed))
end

for (genrand, gv_genrand) in 
    ((:dsfmt_genrand_close1_open2,    :dsfmt_gv_genrand_close1_open2),
     (:dsfmt_genrand_close_open,      :dsfmt_gv_genrand_close_open),
     (:dsfmt_genrand_open_close,      :dsfmt_gv_genrand_open_close),
     (:dsfmt_genrand_open_open,       :dsfmt_gv_genrand_open_open))
    @eval begin
     
        function ($genrand)(s::DSFMT_state)
            ccall(($(string(genrand)),:librandom),
                  Float64,
                  (Ptr{Void},),
                  s.val)
        end

        function ($gv_genrand)()
            ccall(($(string(gv_genrand)),:librandom),
                  Float64,
                  ())
        end
        
    end
end

for (genrand_fill, gv_genrand_fill, genrand_fill_name, gv_genrand_fill_name) in
    ((:dsfmt_fill_array_close1_open2,  :dsfmt_gv_fill_array_close1_open2,
      :dsfmt_fill_array_close1_open2!, :dsfmt_gv_fill_array_close1_open2!),
     (:dsfmt_fill_array_close_open,    :dsfmt_gv_fill_array_close_open,
      :dsfmt_fill_array_close_open!,   :dsfmt_gv_fill_array_close_open!),
     (:dsfmt_fill_array_open_close,    :dsfmt_gv_fill_array_open_close,
      :dsfmt_fill_array_open_close!,   :dsfmt_gv_fill_array_open_close!),
     (:dsfmt_fill_array_open_open,     :dsfmt_gv_fill_array_open_open,
      :dsfmt_fill_array_open_open!,    :dsfmt_gv_fill_array_open_open!))
    @eval begin

        function ($genrand_fill_name)(s::DSFMT_state, A::Array{Float64})
            n = length(A)
            if n <= dsfmt_min_array_size
                for i = 1:n
                    A[i] = rand()
                end
            else
                ccall(($(string(genrand_fill)),:librandom),
                      Void,
                      (Ptr{Void}, Ptr{Float64}, Int),
                      s.val, A, n & 0xfffffffffffffffe)
                if isodd(n)
                    A[n] = rand()
                end
            end
            return A
        end
        
        function ($gv_genrand_fill_name)(A::Array{Float64})
            n = length(A)
            if n <= dsfmt_min_array_size
                for i = 1:n
                    A[i] = rand()
                end
            else
                ccall(($(string(gv_genrand_fill)),:librandom),
                      Void,
                      (Ptr{Void}, Int),
                      A, n & 0xfffffffffffffffe)
                if isodd(n)
                    A[n] = rand()
                end
            end
            return A
        end
    end
end

function dsfmt_genrand_uint32(s::DSFMT_state)
    ccall((:dsfmt_genrand_uint32,:librandom), 
          Uint32,
          (Ptr{Void},),
          s.val)
end

function dsfmt_gv_genrand_uint32()
    ccall((:dsfmt_gv_genrand_uint32,:librandom), 
          Uint32,
          ())
end

## randmtzig

EMANTISSA = 4503599627370496 # /* 52 bit mantissa */
NMANTISSA = 2251799813685248

ZIGGURAT_TABLE_SIZE = 256

ZIGGURAT_NOR_R = 3.6541528853610088
ZIGGURAT_NOR_INV_R = 0.27366123732975828
NOR_SECTION_AREA = 0.00492867323399

ZIGGURAT_EXP_R = 7.69711747013104972
ZIGGURAT_EXP_INV_R = 0.129918765548341586
EXP_SECTION_AREA = 0.0039496598225815571993

# Tables for normal variates
const ki = Array(Uint64, ZIGGURAT_TABLE_SIZE)
const wi = Array(Float64, ZIGGURAT_TABLE_SIZE)
const fi = Array(Float64, ZIGGURAT_TABLE_SIZE)
const pki = pointer(ki)
const pwi = pointer(wi)
const pfi = pointer(fi)
# Tables for exponential variates
const ke = Array(Uint64, ZIGGURAT_TABLE_SIZE)
const we = Array(Float64, ZIGGURAT_TABLE_SIZE)
const fe = Array(Float64, ZIGGURAT_TABLE_SIZE)
const pke = pointer(ke)
const pwe = pointer(we)
const pfe = pointer(fe)

function randmtzig_fill_ziggurat_tables() # Operates on the global arrays
    wib = big(wi)
    fib = big(fi)
    web = big(we)
    feb = big(fe)
    # Ziggurat tables for the normal distribution
    x1 = big(ZIGGURAT_NOR_R)
    wib[256] = x1/NMANTISSA
    fib[256] = exp(-0.5*x1*x1)
    # Index zero is special for tail strip, where Marsaglia and Tsang
    # defines this as
    # k_0 = 2^31 * r * f(r) / v, w_0 = 0.5^31 * v / f(r), f_0 = 1,
    # where v is the area of each strip of the ziggurat.
    ki[1] = uint(itrunc(x1*fib[256]/big(NOR_SECTION_AREA)*NMANTISSA))
    wib[1] = big(NOR_SECTION_AREA)/fib[256]/NMANTISSA
    fib[1] = one(BigFloat)

    for i = 255:-1:2
        # New x is given by x = f^{-1}(v/x_{i+1} + f(x_{i+1})), thus
        # need inverse operator of y = exp(-0.5*x*x) -> x = sqrt(-2*ln(y))
        x = sqrt(-2.0*log(big(NOR_SECTION_AREA)/x1 + fib[i+1]))
        ki[i+1] = uint64(itrunc(x/x1*NMANTISSA))
        wib[i] = x/NMANTISSA
        fib[i] = exp(-0.5*x*x)
        x1 = x
    end

    ki[2] = uint64(0)

    # Zigurrat tables for the exponential distribution
    x1 = big(ZIGGURAT_EXP_R)
    web[256] = x1/EMANTISSA
    feb[256] = exp(-x1)

    # Index zero is special for tail strip, where Marsaglia and Tsang
    # defines this as
    # k_0 = 2^32 * r * f(r) / v, w_0 = 0.5^32 * v / f(r), f_0 = 1,
    # where v is the area of each strip of the ziggurat.
    ke[1] = uint64(itrunc(x1*feb[256]/big(EXP_SECTION_AREA)*EMANTISSA))
    web[1] = big(EXP_SECTION_AREA)/feb[256]/EMANTISSA
    feb[1] = one(BigFloat)

    for i = 255:-1:2
        # New x is given by x = f^{-1}(v/x_{i+1} + f(x_{i+1})), thus
        # need inverse operator of y = exp(-x) -> x = -ln(y)
        x = -log(big(EXP_SECTION_AREA)/x1 + feb[i+1])
        ke[i+1] = uint64(itrunc(x/x1*EMANTISSA))
        web[i] = x/EMANTISSA
        feb[i] = exp(-x)
        x1 = x
    end
    ke[2] = zero(Uint64)

    wi[:] = float64(wib)
    fi[:] = float64(fib)
    we[:] = float64(web)
    fe[:] = float64(feb)
    return nothing
end
randmtzig_fill_ziggurat_tables()
 
import Base.LibRandom.dsfmt_gv_genrand_close1_open2
import Base.LibRandom.dsfmt_genrand_close1_open2
import Base.LibRandom.dsfmt_gv_genrand_open_open
import Base.LibRandom.dsfmt_genrand_open_open

@eval begin
function randmtzig_randn()
    @inbounds begin
        while true
            # arbitrary mantissa (selected by randi, with 1 bit for sign) */
            r = reinterpret(Uint64,dsfmt_gv_genrand_close1_open2()) & 0x000fffffffffffff
            rabs = int64(r>>1)
            idx = rabs & 0xFF
            x = (r&1 != 0x0000000000000000 ? -rabs : rabs)*wi[idx+1]
            
            if rabs < ki[idx+1]
                return x # 99.3% of the time we return here 1st try
            elseif idx == 0
                # As stated in Marsaglia and Tsang
                # For the normal tail, the method of Marsaglia[5] provides:
                # generate x = -ln(U_1)/r, y = -ln(U_2), until y+y > x*x,
                # then return r+x. Except that r+x is always in the positive
                # tail!!!! Any thing random might be used to determine the
                # sign, but as we already have r we might as well use it
                # [PAK] but not the bottom 8 bits, since they are all 0 here!
                while true
                    xx = -$(ZIGGURAT_NOR_INV_R)*log(rand())
                    yy = -log(rand())
                    if yy+yy > xx*xx
                        return (rabs & 0x100) != 0x0000000000000000 ? -$(ZIGGURAT_NOR_R)-xx : $(ZIGGURAT_NOR_R)+xx
                    end
                end
            elseif (fi[idx] - fi[idx+1])*rand() + fi[idx+1] < exp(-0.5*x*x)
                return x
            end
        end
    end
end
function randmtzig_randn(s::DSFMT_state)
    @inbounds begin
        while true
            # arbitrary mantissa (selected by randi, with 1 bit for sign) */
            r = reinterpret(Uint64,dsfmt_genrand_close1_open2(s)) & 0x000fffffffffffff
            rabs = int64(r>>1)
            idx = rabs & 0xFF
            x = (r&1 != 0x0000000000000000 ? -rabs : rabs)*wi[idx+1]
            
            if rabs < ki[idx+1]
                return x # 99.3% of the time we return here 1st try
            elseif idx == 0
                # As stated in Marsaglia and Tsang
                # For the normal tail, the method of Marsaglia[5] provides:
                # generate x = -ln(U_1)/r, y = -ln(U_2), until y+y > x*x,
                # then return r+x. Except that r+x is always in the positive
                # tail!!!! Any thing random might be used to determine the
                # sign, but as we already have r we might as well use it
                # [PAK] but not the bottom 8 bits, since they are all 0 here!
                while true
                    xx = -$(ZIGGURAT_NOR_INV_R)*log(rand(s))
                    yy = -log(rand(s))
                    if yy+yy > xx*xx
                        return (rabs & 0x100) != 0x0000000000000000 ? -$(ZIGGURAT_NOR_R)-xx : $(ZIGGURAT_NOR_R)+xx
                    end
                end
            elseif (fi[idx] - fi[idx+1])*rand(s) + fi[idx+1] < exp(-0.5*x*x)
                return x
            end
        end
    end
end

function randmtzig_exprnd()
    @inbounds begin
        while true
            ri = reinterpret(Uint64,dsfmt_gv_genrand_close1_open2()) & 0x000fffffffffffff
            idx = ri & 0xFF
            # x = ri*we[idx+1]
            x = ri*unsafe_load(pwe, idx+1)
            if ri < unsafe_load(pke, idx+1)
                return x # 98.9% of the time we return here 1st try
            elseif idx == 0
            # As stated in Marsaglia and Tsang
            # For the exponential tail, the method of Marsaglia[5] provides:
            # x = r - ln(U)
                x = $(ZIGGURAT_EXP_R) - log(rand())
            elseif (unsafe_load(pfe, idx) - unsafe_load(pfe, idx+1))*rand() + unsafe_load(pfe, idx+1) < exp(-x)
                return x
            end
        end
    end
end
function randmtzig_exprnd(s::DSFMT_state)
    @inbounds begin
        while true
            ri = reinterpret(Uint64,dsfmt_gv_genrand_close1_open2(s)) & 0x000fffffffffffff
            idx = ri & 0xFF
            x = ri*unsafe_load(we, idx+1)
            if ri < unsafe_load(ke, idx+1)
                return x # 98.9% of the time we return here 1st try
            elseif idx == 0
            # As stated in Marsaglia and Tsang
            # For the exponential tail, the method of Marsaglia[5] provides:
            # x = r - ln(U)
                x = $(ZIGGURAT_EXP_R) - log(rand(s))
            elseif (unsafe_load(fe,idx) - unsafe_load(fe,idx+1))*rand(s) + unsafe_load(fe,idx+1) < exp(-x)
                return x
            end
        end
    end
end
end #eval

## Windows entropy

@windows_only begin
    function win32_SystemFunction036!(a::Array{Uint32})
        ccall((:SystemFunction036,:Advapi32),stdcall,Uint8,(Ptr{Void},Uint32),a,length(a)*sizeof(eltype(a)))
    end
end

end # module
