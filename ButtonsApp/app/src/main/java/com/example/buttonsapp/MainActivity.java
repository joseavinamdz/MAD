package com.example.buttonsapp;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Switch;
import android.widget.Toast;
import android.widget.ToggleButton;

public class MainActivity extends AppCompatActivity {

    private RadioGroup radioGroup;
    private CheckBox checkBoxGod;
    private CheckBox checkBoxRegular;
    private CheckBox checkBoxBad;
    private ToggleButton toggleButton;
    private Button openActivityButton;
    private EditText editTextName, editTextEmail;

    private Bundle bundle;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //CheckBoxes Listeners
        checkBoxGod = findViewById(R.id.checkBox_good);
        checkBoxRegular = findViewById(R.id.checkBox_regular);
        checkBoxBad = findViewById(R.id.checkBox_bad);
        toggleButton = findViewById(R.id.toggleButton_on_off);
        openActivityButton = findViewById(R.id.button_open_activity);
        editTextName = findViewById(R.id.editText_name);
        editTextEmail = findViewById(R.id.editText_email);


        //CheckBoxes Listeners
        checkBoxGod.setOnClickListener(view -> System.out.println("Good Checked!"));

        checkBoxRegular.setOnClickListener(view -> System.out.println("Regular Checked!"));

        checkBoxBad.setOnClickListener(view -> {
            if (checkBoxBad.isChecked())
                Toast.makeText(view.getContext(),"Bad Checked!", Toast.LENGTH_SHORT).show();
            else
                Toast.makeText(view.getContext(),"Bad Unchecked", Toast.LENGTH_SHORT).show();
        });


        //RadioGroup Listener
        radioGroup = findViewById(R.id.radio_group);
        radioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            switch (checkedId) {

                case R.id.radioButton_large:
                    Toast.makeText(getApplicationContext(),
                                    "Large, then...",
                                    Toast.LENGTH_SHORT).show();
                    break;

                case R.id.radioButton_medium:
                    Toast.makeText(getApplicationContext(),
                                    "Medium, then...",
                                    Toast.LENGTH_SHORT).show();
                    break;

                case R.id.radioButton_small:
                Toast.makeText(getApplicationContext(),
                        "Small, then...",
                        Toast.LENGTH_SHORT).show();
                break;
            }
        });


        //ToggleButton Listener
        toggleButton.setOnClickListener(myOnClickListener);

        //Button Listener
         openActivityButton.setOnClickListener(new View.OnClickListener() {
             @Override
             public void onClick(View v) {
                 String name = editTextName.getText().toString();
                 String email = editTextEmail.getText().toString();

                 bundle = new Bundle();
                 bundle.putString("Name", name);
                 bundle.putString("Email", email);

                 System.out.println(bundle.getString("Name"));
                 System.out.println(bundle.getString("Email"));

             }
         });


    }


    //ToggleButton Listener's Method

    private void toggleListener(View view) {
        if(((ToggleButton) view).isChecked())
            Toast.makeText(view.getContext(), "Toggle Enabled", Toast.LENGTH_SHORT).show();
        else
            Toast.makeText(view.getContext(), "Toggle Not Enabled", Toast.LENGTH_SHORT).show();
    }

    private View.OnClickListener myOnClickListener = new View.OnClickListener() {
        @Override
        public void onClick(View view) {
            toggleListener(view);
        }
    };


    public void switchListener(View view) {
        if(((Switch)view).isChecked())
            Toast.makeText(view.getContext(), "Switch ON", Toast.LENGTH_SHORT).show();
        else
            Toast.makeText(view.getContext(), "Switch OFF", Toast.LENGTH_SHORT).show();
    }


}
