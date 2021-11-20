import "./Login.scss";
import React, { useState } from 'react';
import { Loader, Title, Input, Form, SubmitButton, Modal } from "./BasicComponents";
import facebookIcon from '../facebook.svg';
import infocsv from '../info.svg';
import blueInfo from '../blueInfo.svg';


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
            
        } else {
            
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
    const infoText = `Skipping the login phase will result in less accurate results, 
    and in some cases will prevent the program from returning a result. Visit the About page for more detail`
    return (
        <div className='screen'>
            <Form>
                <Title title='Welcome to Hebrew Fake News Analayzer!' />
                <div className='image'>
                    <img src={facebookIcon } className='icon' />
                    <label className='label'>Facebook login:</label>
                 
                </div>
                <div className='fieldsLogin'>
                    <Input label='Email' type='text' value={name} name="User name" placeholder="example@gmail.com" onChange={(e) => setName(e.target.value)} />
                    <Input label='Password' type='password' value={password} name="Password" onChange={(e) => setPassword(e.target.value)} />
                    <SubmitButton onSubmit={onSubmit} />
                </div>
                <div  className='buttom'>
                    <a className='link' href="/ScanPost">Skip</a>
                    <img title={infoText} src={blueInfo} className='info' />
                </div>
                <Toggle />
            </Form>
        </div>
    )
}