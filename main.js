import { JSarray,makeobj } from "./JSarray.js";


// following assumptions are made here, 

// data had dim dXn
// labels had dim 1Xn 

// lambda is a regularizer and is a float 

// th had dim (d,)
// th0 is scalar 

// data that is sent to loss_function_gradient will be (d,
//labels will be scalar
function loss_function_grade_th(x, y, th, th0){
    // console.log(th.dotproduct(x))
    return x.mutiply(2*(th.dotproduct(x)+th0-y)) 
}

function loss_function_grade_th0(x, y, th, th0){
    return 2*(th.dotproduct(x)+th0-y)
}

// X is d*n JSarray. 
// Y is 1*n JSarray 
// th is dim (d, )
// th0, lambda are scalars. 
function total_loss_function_grade_th(X, Y, th, th0, lambda){

    let d = X.getsize()[0]
    let n = X.getsize()[1]

    let output = makeobj([d])

    
    for(var i=0; i<n; ++i){
        let new_array= X.allindex(i); 
        output = output.add(loss_function_grade_th(new_array, y.getelement(0).getelement(i), th, th0)) 
    }

    output = output.mutiply((1/n))
    output = output.add(th.mutiply(2*lambda))

    return output

}


function total_loss_function_grade_th0(X, Y, th, th0, lambda){
    let d = X.getsize()[0]
    let n = X.getsize()[1]

    let output = 0 

    for(var i=0; i<n; ++i){
        let new_array= X.allindex(i); 
        output = loss_function_grade_th0(new_array, y.getelement(0).getelement(i), th, th0)
    }

    return output
}

function gd(data, labels, lambda, iterations, eta){
    let d = data.getsize()[0]
    let n = data.getsize()[1]

    th = makeobj([d])
    th0 = 0 
    for(var i=0; i<iterations; ++i){
        // console.log(total_loss_function_grade_th(data, labels, th, th0, lambda))
        th = th.add(total_loss_function_grade_th(data, labels, th, th0, lambda).mutiply(-eta))
        th0 += -eta*total_loss_function_grade_th0(data, labels, th, th0, lambda)
    }
    return [th, th0]
}



let X = new JSarray([[1,2,3],[4,5,6]])
let th = new JSarray([1,1])
let th0 = 0 
let y = new JSarray([[1, 1, 1]])



export default {gd}