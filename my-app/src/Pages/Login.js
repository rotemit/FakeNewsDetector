import './Login.css'
import React, { useState } from 'react';
import { LoginForm } from '../Components/LoginForm/LoginForm';


export const Login = () => {
    const adminUser = {
        email: "admin@admin.com",
        password: "12345"
    }
    const [user, setUser] = useState({email:"", password: ""});
    const [error, setError] = useState("");

    const Login_func = details => {
        console.log(details);

        if(details.email == adminUser.email &&  details.password == adminUser.password) {
            console.log("Looged in");
            setUser({
                email: details.email,
                password: details.password
            });
        } else {
            console.log("Details do not match")
        }
    
    }

    const Logout = () => {
        console.log("Logout");
    }
    
    return(

        <div className="App">
           {(user.email != "") ? (
               <div className="welcome">
                   <h2>Welcome, <span>{user.name}</span></h2>
                   <button>Logout</button>
                </div>
           ) : (
               <LoginForm Login={Login_func} error={error}/>
           )}  
        </div>
    )
    }