// import React from 'react';
import "./Login.css";
import React, { useEffect, useState } from 'react';


export const Login = () => {
    const [name , setName] = useState('');
    const [password , setPassword] = useState('');

    async function onSubmit()  {   
        setName('');
        setPassword('');
        const response = await fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name: name, password: password }),
        })
        if (response.ok) {
            response.json().then((res) => console.log(res)).catch((e) => console.log(e))
        } 
    
    }
 
    return (
      <div className='screen'>
        <div className='form'>
        <h1>Welcome to our app!</h1>
        <div className='fields'>
            <div class="field">
                <label>User name</label>
                <input type="text" name="User name" placeholder="example@gmail.com" onChange={(e) => setName(e.target.value)} />
            </div>
            <div class="field">
                <label>Password</label>
                <input type="password" name="Password" onChange={(e) => setPassword(e.target.value)} />
            </div>
            <button class="ui button" type="submit" onClick={onSubmit}>Submit</button>
        </div>
            <a class="item" href="/ScanPost">Skip</a>
        </div>
      </div>
    )
}