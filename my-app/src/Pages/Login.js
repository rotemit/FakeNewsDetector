// import React from 'react';
import "./Login.scss";
import React, { useState } from 'react';
import { Loader, Title, Input, Form, SubmitButton, Modal } from "./BasicComponents";


export const Login = () => {
    const [name , setName] = useState('');
    const [password , setPassword] = useState('');
    const [isDone, setIsDone] = useState(false);
    const [isValid, setIsValid] = useState(false);
    const [showModal, setShowModal] = useState(false);
    const [isSubmit, setIsSubmit] = useState(false);


    async function onSubmit()  {  
        setIsSubmit(true);
        setIsDone(false); 
        const response = await fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name: name, password: password }),
        })
        if (response.ok) {
            
            setIsDone(true);
            response.json().then((res) => setIsValid(res.result));
            if (!isValid) {
                setShowModal(true);                
            }  
            else {
                window.location.href ='/ScanPost';
            }
            
            setName('');
            setPassword('');
            
        } 
    
    }

    const clickCloseModal = () => {
        setShowModal(false);
        setIsSubmit(false);
        console.log('clicked!');
    }

    const Toggle = () => {
        if (isSubmit) {
            if (isDone && isValid) {
                window.location.href ='/ScanPost';
                return <p>success</p>;
            } else if (isDone && !isValid) {
                return (
                <Modal handleClose={clickCloseModal} show={showModal} text={`userName or password are not correct. please try again`} />
                )
            } else {
                return <Loader />
            }
        } 
        return <div></div>
    }
 
    return (
        <div className='screen'>
            <Form>
                <Title title='Welcome to our app!' />
                <div className='fields'>
                    <Input label='User name' type='text' value={name} name="User name" placeholder="example@gmail.com" onChange={(e) => setName(e.target.value)} />
                    <Input label='Password' type='password' value={password} name="Password" onChange={(e) => setPassword(e.target.value)} />
                    <SubmitButton onSubmit={onSubmit} />
                </div>
                <a className='link' href="/ScanPost">Skip</a>
                <Toggle />
            </Form>
        </div>
    )
}