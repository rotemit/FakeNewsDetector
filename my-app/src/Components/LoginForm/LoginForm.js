import React, {useState} from "react"
import "./LoginForm.css"

export const LoginForm = ({Login_func, error}) => {
    const [details, setDetails] = useState({email:"", password:""});

    const submitHandler = e => {
        e.preventDefault();
        Login_func(details);
    }


    return (
        <form onSubmit={submitHandler}>
            <div className="form-inner">
                <h2>Login</h2>
                {/* {Error!} */}
                <div className="form-group">
                <label htmlFor="email">Email:</label>
                    <input type="email" name="email" id="email" onChange={e => setDetails({...details, email: e.target.value})} />
                </div>
                <div className="form-group">
                    <label htmlFor="password">password:</label>
                    <input type="password" name="password" id="password" onChange={e => setDetails({...details, password: e.target.value})}/>
                </div>
                <input type="submit" value="LOGIN"/>
            </div>
        </form>
    )
}